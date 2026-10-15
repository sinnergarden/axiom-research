"""PIT financial reports materialized as explicitly derived daily input leaves.

Data selects revisions and computes financial bases. Research selects economic
report periods and stores typed inputs. Core remains the Feature executor.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .stock_artifacts import digest, write_json
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import instant_us, write_buffer, write_part

POLICY = 'latest_visible_report_strict_before_local_session_v1'
DOMAIN = 'materialized_financial_events'
FORECAST_POLICY='latest_visible_announcement_then_report_period_v1'
FORECAST_CODES={'预增':1,'略增':2,'扭亏':3,'续盈':4,'预减':5,'略减':6,'首亏':7,'续亏':8}


def _period_back(period, quarters):
    year,month = int(period[:4]),int(period[5:7])
    require(period[5:] in ('03-31','06-30','09-30','12-31'),'quarter-end report required')
    ordinal = year*4 + month//3-1-quarters
    year,q = divmod(ordinal,4)
    return f'{year:04d}-'+('03-31','06-30','09-30','12-31')[q]


def _cutoff(day, actual, zone):
    boundary = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc)
    strict = boundary-timedelta(microseconds=1)
    original = datetime.fromisoformat(actual.replace('Z','+00:00'))
    require(original.tzinfo is not None,'timezone-aware source cutoff required')
    return min(strict,original).isoformat()


def _projection_valid(spec):
    if spec.get('derivation') in ('report_observation','report_announcement_ordinal','forecast_category_code'):
        return type(spec.get('reports_back')) is int and spec['reports_back']>=0
    return (spec.get('derivation') in ('ttm','single_quarter','reported') and
            type(spec.get('quarters_back')) is int and spec['quarters_back']>=0)


def validate_financial_materializations(manifest, child=None):
    """Admission hook: derived declarations and every original component clock."""
    declarations = manifest.get('financial_materializations',{})
    require(type(declarations) is dict,'financial materialization map required')
    require(set(declarations)=={f for f,d in manifest.get('daily_fields',{}).items() if d['domain']==DOMAIN},
            'financial declarations must cover every derived field')
    for field,spec in declarations.items():
        coverage=spec.get('coverage_security_ids')
        require(type(coverage) is list and bool(coverage) and coverage==sorted(set(coverage)) and
                set(coverage)<=set(manifest['spec']['universe']),'financial coverage must be explicit sorted source subset')
        require(type(spec.get('event_snapshot')) is str and spec['event_snapshot'] not in ('','current','latest'),
                'financial Snapshot must be pinned')
        require(field in manifest.get('daily_fields',{}) and spec['selection'] in (POLICY,FORECAST_POLICY) and
                _projection_valid(spec) and (spec['derivation'] in ('reported','report_observation','report_announcement_ordinal','forecast_category_code') or spec['basis']=='cumulative_ytd'), 'financial materialization declaration')
        require(manifest['daily_fields'][field]=={'domain':DOMAIN,'dtype':'float64','unit':spec['unit']},
                'financial materialization typed leaf mismatch')
    if child is None: return
    financial = set(child['fields']) & set(declarations)
    if not financial: return
    require(child['context'].get('domain')==DOMAIN and child['context'].get('selection') in (POLICY,FORECAST_POLICY),
            'financial evidence selection required')
    for field in financial:
        declaration=declarations[field]
        for row in child['field_meta'][field]['by_key']:
            require(row['security_id'] in declaration['coverage_security_ids'] and
                    child['context']['snapshot_id']==declaration['event_snapshot'],'financial evidence source/coverage mismatch')
            cutoff=_cutoff(row['session'],manifest['spec']['cutoff_by_session'][row['session']],declaration['session_timezone'])
            require(row['query_cutoff']==cutoff and row['selection']==declaration['selection'],'financial evidence cutoff mismatch')
            if declaration['derivation'] in ('report_observation','report_announcement_ordinal','forecast_category_code'):
                periods=row['visible_report_periods'];offset=declaration['reports_back']
                require(len(periods)==len(set(periods)) and (declaration['selection']==FORECAST_POLICY or periods==sorted(periods)) and row['report_period']==(periods[-1-offset] if len(periods)>offset else None),'distinct visible report selection mismatch')
            else:
                require(row['report_period'] is None or row['report_period']==_period_back(row['current_report_period'],declaration['quarters_back']),
                        'financial lag must use economic report quarters')
            clocks=[]
            for component in row['components']:
                provenance=component['provenance']
                if provenance is None: continue
                require(provenance.get('revision_id') and provenance.get('usable_from'), 'financial component revision/clock required')
                require(instant_us(provenance['usable_from'])<=instant_us(cutoff),'future financial component')
                clocks.append(provenance['usable_from'])
            expected=max(clocks,key=instant_us) if clocks else cutoff
            require(row['usable_from']==expected,'financial component max clock mismatch')


def extend_financial_feature_source(data, *, source, query, field, outputs, unit,
        destination, event_snapshot, session_timezone='Asia/Shanghai', block_sessions=20,
        maximum_added_bytes=64*1024**2, maximum_event_cache_bytes=64*1024**2, progress=None):
    """Prepare a financial batch with one pinned Data owner across all blocks."""
    from axiom_data import EventReadOwner
    with EventReadOwner(data, snapshot=event_snapshot,
                       maximum_cache_bytes=maximum_event_cache_bytes) as owner:
        return _extend_financial_feature_source(data, source=source, query=query,
            field=field, outputs=outputs, unit=unit, destination=destination,
            event_snapshot=event_snapshot, session_timezone=session_timezone,
            block_sessions=block_sessions, maximum_added_bytes=maximum_added_bytes,
            maximum_event_cache_bytes=maximum_event_cache_bytes, progress=progress,
            event_owner=owner)


def _extend_financial_feature_source(data, *, source, query, field, outputs, unit,
        destination, event_snapshot, session_timezone='Asia/Shanghai', block_sessions=20,
        maximum_added_bytes=64*1024**2, maximum_event_cache_bytes=64*1024**2, progress=None, event_owner=None):
    """Append reported/TTM/single-quarter leaves without rebuilding existing columns.

    ``outputs`` maps leaf name to {derivation, quarters_back}. Every leaf uses
    the latest visible current report and an explicit fiscal-quarter offset.
    A bounded block shares original event partitions across all daily cutoffs
    and both current/prior-year derivations. Revision selection is Data-owned.
    Missing/retracted components remain null; original statuses/revisions are
    retained in cell evidence, with generic source_missing in the packed ABI.
    This version accepts the saved source's best_effort_vendor_v1 policy only.
    This writes only destination. It does not publish a Data Snapshot, fetch
    suppliers, calculate a Feature, or claim historical coverage/performance.
    """
    import numpy as np
    from axiom_data import EventQuery, iter_event_cutoffs, single_quarter, ttm
    from axiom_data.financial import FINANCIAL_RECIPE_VERSION
    from .stock_feature_columns import FeatureColumnSource, REASONS
    require(isinstance(query,EventQuery) and query.domain in ('financial_events','cash_flow_events','balance_sheet_events','financial_indicator_events','shareholder_count_events','top_holders_reports','forecast_events') and
            query.time_field=='report_period' and query.fields==(field,) and query.purpose=='decision_facts',
            'one explicit financial event field required')
    selection=FORECAST_POLICY if query.domain=='forecast_events' else POLICY
    manifest=deepcopy(source.manifest); spec=manifest['spec']; days=spec['read_sessions']; symbols=spec['universe']
    require(bool(query.symbols) and list(query.symbols)==sorted(set(query.symbols)) and
            set(query.symbols)<=set(symbols) and query.pit_policy==spec['pit_policy']=='best_effort_vendor_v1','financial source scope/policy mismatch')
    require(query.filters.get('endpoint') or query.domain in ('shareholder_count_events','top_holders_reports','forecast_events'),'financial stream endpoint required')
    require(type(block_sessions) is int and 0<block_sessions<=32,'one to 32 financial sessions per block')
    require(type(outputs) is dict and outputs and not set(outputs)&set(source.physical_fields()),'new financial outputs required')
    for name,projection in outputs.items():
        require(name.isidentifier() and set(projection)==({'derivation','reports_back'} if projection.get('derivation') in ('report_observation','report_announcement_ordinal','forecast_category_code') else {'derivation','quarters_back'}) and _projection_valid(projection),'explicit financial output recipe required')
    require(event_snapshot not in ('','current','latest'),'pinned independent financial Snapshot required')
    native=event_owner.manifest['domains'][query.domain]
    actual=native['contract']['fields'][field]
    require((actual.get('unit')==unit or actual.get('dtype')=='string' and unit=='dimensionless' and all(p['derivation']=='forecast_category_code' for p in outputs.values()) or actual.get('dtype')=='date' and unit=='days' and all(p['derivation']=='report_announcement_ordinal' for p in outputs.values())) and (all(p['derivation'] in ('reported','report_observation','report_announcement_ordinal','forecast_category_code') for p in outputs.values()) or actual.get('basis')=='cumulative_ytd'),'financial input unit/basis mismatch')
    shape=(len(days),len(symbols)); required=len(days)*len(symbols)*27*len(outputs)
    require(type(maximum_added_bytes) is int and required<=maximum_added_bytes,'financial output buffer budget exceeded')
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    from .feature_input_buffers import allocate_feature_buffer, seal_feature_buffer
    arrays=dict(source.arrays)
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
           'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
           'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    declarations={name:{**projection,'source_domain':query.domain,'source_field':field,
        'endpoint':query.filters.get('endpoint'),'report_type':query.filters.get('report_type'),'basis':actual.get('basis','reported'),
        'unit':unit,'selection':selection,'session_timezone':session_timezone,'recipe_version':FINANCIAL_RECIPE_VERSION,
        'report_start':query.start,'report_end':query.end,'event_snapshot':event_snapshot,
        'coverage_security_ids':list(query.symbols)} for name,projection in outputs.items()}
    chunks=[]; metrics={'event_blocks':0,'event_cutoff_selections':0,'existing_field_reads':0,'added_fields':len(outputs)}
    from .feature_input_checkpoint import extension_checkpoint, atomic_json
    from .feature_input_evidence import check_input_disk
    staging,arrays=extension_checkpoint(root,source=source,fields=outputs,kinds=kinds,sessions=days,
        block_sessions=block_sessions,binding={'declarations':declarations},metrics=metrics)
    chunks=list(staging.state['chunks']);metrics=dict(staging.state['metrics'])
    for start in range(staging.state['completed_dates'],len(days),block_sessions):
        check_input_disk(root,min(block_sessions,len(days)-start)*sum(a.shape[1]*a.dtype.itemsize for a in staging.arrays.values()))
        selected=days[start:start+block_sessions]; metadata={n:{'dtype':'float64','unit':unit,'by_key':[]} for n in outputs}
        contexts=[]
        for si in range(0,len(query.symbols),256):
            subset=list(query.symbols[si:si+256])
            cutoffs=[_cutoff(d,spec['cutoff_by_session'][d],session_timezone) for d in selected]
            batches=iter_event_cutoffs(data,snapshot=event_snapshot,query=replace(query,symbols=tuple(subset),fields=tuple(dict.fromkeys((field,'announcement_date'))) if query.domain=='forecast_events' else query.fields),
                cutoffs=cutoffs,maximum_cache_bytes=maximum_event_cache_bytes,owner=event_owner)
            metrics['event_blocks']+=1
            for offset,(day,batch) in enumerate(zip(selected,batches)):
                metrics['event_cutoff_selections']+=1
                require(batch.context['snapshot_id']==event_snapshot and
                        instant_us(batch.context['query']['cutoff'])==instant_us(cutoffs[offset]),'financial selected batch binding')
                contexts.append(deepcopy(batch.context))
                latest={}
                for row in batch.frame.to_dict('records'):
                    require(row['security_id'] in subset and ('endpoint' not in query.filters or row.get('endpoint')==query.filters['endpoint']) and
                            ('report_type' not in query.filters or row.get('report_type')==query.filters['report_type']),'unexpected financial event stream')
                    if query.domain=='forecast_events' or row['report_period']<=day:
                        latest[row['security_id']]=max(latest.get(row['security_id'],row['report_period']),row['report_period'])
                # Prepare each Data derivation once for the union of current
                # and prior-year target periods, not once per output leaf.
                event_rows=batch.frame.to_dict('records')
                ordered={security:[] for security in subset}
                for record in event_rows:
                    if query.domain=='forecast_events' or record['report_period']<=day:
                        ordered[record['security_id']].append(record)
                visible_periods={security:[r['report_period'] for r in sorted(records,key=lambda r:(r['announcement_date'],r['report_period']) if query.domain=='forecast_events' else (r['report_period'],))] for security,records in ordered.items()}
                targets_by_name={}
                for name,projection in outputs.items():
                    if projection['derivation'] in ('report_observation','report_announcement_ordinal','forecast_category_code'):
                        offset_back=projection['reports_back']
                        targets_by_name[name]={s:periods[-1-offset_back] for s,periods in visible_periods.items() if len(periods)>offset_back}
                    else:
                        targets_by_name[name]={s:_period_back(p,projection['quarters_back']) for s,p in latest.items()}
                derived_by_recipe={}
                for recipe in {p['derivation'] for p in outputs.values()}:
                    periods=sorted({p for name,target in targets_by_name.items()
                                    if outputs[name]['derivation']==recipe for p in target.values()})
                    if recipe in ('reported','report_observation','report_announcement_ordinal','forecast_category_code'):
                        records={(r['security_id'],r['report_period']):r for r in batch.frame.to_dict('records')}
                        by={}
                        for item in batch.field_meta[field]['by_key']:
                            key=(item['security_id'],item['report_period'])
                            by[key]={'status':item['status'],'missing_reason':item.get('missing_reason'),
                                     'components':[{'security_id':key[0],'report_period':key[1],
                                         'status':item['status'],'provenance':deepcopy(item)}]}
                        derived_by_recipe[recipe]=(records,by)
                        continue
                    derived=(ttm if recipe=='ttm' else single_quarter)(batch,field=field,
                        endpoint=query.filters['endpoint'],report_type=query.filters['report_type'],basis='cumulative_ytd',unit=unit,
                        periods=periods) if periods else None
                    records={} if derived is None else {(r['security_id'],r['report_period']):r for r in derived.to_json()['records']}
                    by={} if derived is None else {(r['security_id'],r['report_period']):r for r in derived.field_meta[field]['by_key']}
                    derived_by_recipe[recipe]=(records,by)
                for name,projection in outputs.items():
                    targets=targets_by_name[name]
                    records,by=derived_by_recipe[projection['derivation']]
                    for sj,security in enumerate(subset):
                        period=targets.get(security); meta=deepcopy(by.get((security,period),{'components':[],'status':'source_missing','missing_reason':'no_visible_report'}))
                        value=records.get((security,period),{}).get(field);valid=meta['status']=='value' and value is not None
                        if valid and projection['derivation']=='report_announcement_ordinal':
                            from datetime import date
                            value=float(date.fromisoformat(str(value)[:10]).toordinal())
                        elif valid and projection['derivation']=='forecast_category_code':
                            value=float(FORECAST_CODES.get(value,0))
                        else:
                            value=float(value) if valid else None
                        clocks=[c['provenance']['usable_from'] for c in meta['components'] if c['provenance'] is not None]
                        clock=max(clocks,key=instant_us) if clocks else cutoffs[offset]
                        bases=sorted({c['provenance']['availability_basis'] for c in meta['components'] if c['provenance'] is not None})
                        evidence={'security_id':security,'session':day,'report_period':period,'current_report_period':latest.get(security),
                            'selection':selection,'query_cutoff':cutoffs[offset],'status':meta['status'],
                            'financial_missing_reason':meta['missing_reason'],'missing_reason':None if valid else 'source_missing',
                            'usable_from':clock,'availability_basis':bases[0] if len(bases)==1 else 'derived_component_max' if bases else 'no_visible_report_at_query_cutoff',
                            'component_availability_bases':bases,
                            'components':meta['components'],'selected_value':value,'revision_id':digest(meta)}
                        if projection['derivation'] in ('report_observation','report_announcement_ordinal','forecast_category_code'):evidence['visible_report_periods']=visible_periods[security]
                        metadata[name]['by_key'].append(evidence)
                        position=(start+offset,source.security_positions[security])
                        for kind,v in {'values':value if valid else 0,'original_values':value if valid else 0,
                            'validity':valid,'available':instant_us(clock),'available_validity':True,
                            'reasons':0 if valid else REASONS.index('source_missing')}.items():arrays[name+'::'+kind][position]=v
        child=seal({'contract_version':'reader_column_evidence_v1','context':{'domain':DOMAIN,'selection':selection,
            'snapshot_id':event_snapshot,'event_contexts':contexts},'field_meta':metadata,'fields':list(outputs),'sessions':selected},'evidence_ref')
        check_manifest={**manifest,'daily_fields':{**manifest.get('daily_fields',{}),**{n:{'domain':DOMAIN,'dtype':'float64','unit':unit} for n in outputs}},
                        'financial_materializations':{**manifest.get('financial_materializations',{}),**declarations}}
        validate_financial_materializations(check_manifest,child)
        chunks.append({'sessions':selected,'evidence':[write_part(root,child,'evidence_ref')]})
        staging.commit(start+len(selected),chunks,metrics)
        if progress is not None: progress({'completed_dates':start+len(selected),'total_dates':len(days),**metrics})
    for name in outputs:
        for kind,(_,dtype) in kinds.items():
            value=arrays[name+'::'+kind]
            manifest['buffers'][name+'::'+kind],arrays[name+'::'+kind]=seal_feature_buffer(root,value,dtype=dtype,shape=shape,retain_staging=True)
    manifest.setdefault('daily_fields',{}).update({n:{'domain':DOMAIN,'dtype':'float64','unit':unit} for n in outputs})
    manifest.setdefault('financial_materializations',{}).update(declarations)
    manifest.setdefault('daily_extensions',[]).append({'fields':list(outputs),'chunks':chunks})
    profile={'contract':{'contract_id':'research.financial_materialization.v1','logical_key':['security_id','session'],
                        'fields':{n:{'dtype':'float64','unit':unit} for n in outputs}},
             'source_profile':{'id':'research.financial_materialization.v1','financial_inputs':{k:native[k] for k in ('contract','source_profile')},
                'projections':{n:{k:v for k,v in d.items() if k not in ('event_snapshot','report_start','report_end','coverage_security_ids')} for n,d in declarations.items()},'selection':selection,'not_native_daily':True}}
    previous=manifest['source_profiles'].get(DOMAIN)
    if previous is not None:
        merged=deepcopy(previous)
        owner=merged['source_profile']
        by_field=owner.setdefault('financial_inputs_by_field',{
            name:deepcopy(owner['financial_inputs']) for name in owner['projections']})
        by_field.update({name:deepcopy(profile['source_profile']['financial_inputs']) for name in outputs})
        merged['contract']['fields'].update(profile['contract']['fields'])
        merged['source_profile']['projections'].update(profile['source_profile']['projections'])
        profile=merged
    manifest['source_profiles'][DOMAIN]=profile
    validate_financial_materializations(manifest)
    manifest.pop('source_ref');manifest=seal(manifest,'source_ref');atomic_json(root/'source.json',manifest)
    staging.retire(manifest)
    metrics['event_reader']=dict(event_owner.metrics)
    return FeatureColumnSource(manifest,arrays),metrics

"""PIT financial reports materialized as explicitly derived daily input leaves.

Data selects revisions and computes financial bases. Research selects economic
report periods and stores typed inputs. Core remains the Feature executor.
"""
from copy import deepcopy
from collections import OrderedDict
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .stock_artifacts import digest, write_json
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import instant_us, write_buffer, write_part

POLICY = 'latest_visible_report_strict_before_local_session_v1'
QUARTER_POLICY = 'latest_visible_calendar_quarter_strict_before_local_session_v2'
DOMAIN = 'materialized_financial_events'
FORECAST_POLICY='latest_visible_announcement_then_report_period_v1'
FORECAST_CODES={'预增':1,'略增':2,'扭亏':3,'续盈':4,'预减':5,'略减':6,'首亏':7,'续亏':8}


def _quarter_projection(batch, event_rows, tokens, day):
    """Project already PIT-selected revisions; never select an older nonnull value.

    Native observations remain intact. Quarter anchors, formula inputs and
    dependency tokens share this exact eligible set, including missing fields.
    """
    from axiom_data import DataBatch
    rows=[r for r in event_rows if r['report_period']<=day and
          r['report_period'][5:] in ('03-31','06-30','09-30','12-31')]
    keys=batch.context['logical_key']
    wanted={tuple(str(r[k]) for k in keys) for r in rows}
    if len(rows)==len(event_rows):
        frame=batch.frame;meta=batch.field_meta
    else:
        mask=[tuple(str(r[k]) for k in keys) in wanted for r in event_rows]
        frame=batch.frame.loc[mask]
        meta={field:{**value,'by_key':[r for r in value['by_key']
              if tuple(str(r[k]) for k in keys) in wanted]} for field,value in batch.field_meta.items()}
    selected_tokens={s:{k:v for k,v in values.items() if k in wanted} for s,values in tokens.items()}
    return DataBatch(frame,meta,{**batch.context,'financial_selection':QUARTER_POLICY}),rows,selected_tokens


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


def _dependencies(period,recipe):
    """Identity only; arithmetic remains in Data's original financial recipes."""
    if period is None:return ()
    if recipe not in ('ttm','single_quarter'):return (period,)
    needed=[]
    for back in range(4 if recipe=='ttm' else 1):
        quarter=_period_back(period,back);needed.append(quarter)
        if quarter[5:7]!='03':needed.append(_period_back(quarter,1))
    return tuple(dict.fromkeys(needed))


def _projection_targets(rows, outputs, symbols, day, *, forecast=False):
    ordered={s:[] for s in symbols}
    for row in rows:
        if forecast or row['report_period']<=day:ordered[row['security_id']].append(row)
    periods={s:[r['report_period'] for r in sorted(records,key=lambda r:
        (r['announcement_date'],r['report_period']) if forecast else (r['report_period'],))]
        for s,records in ordered.items()}
    targets={}
    for name,projection in outputs.items():
        if 'reports_back' in projection:
            require(all(len(p)==len(set(p)) for p in periods.values()),'distinct visible report periods required')
            back=projection['reports_back']
            targets[name]={s:p[-1-back] for s,p in periods.items() if len(p)>back}
        else:
            targets[name]={s:_period_back(max(p),projection['quarters_back']) for s,p in periods.items() if p}
    return targets


class _FinancialMemo:
    """One field/stream/Snapshot/rule owner; keys contain actual dependency tokens."""
    def __init__(self,maximum_bytes=128*1024**2):
        self.entries=OrderedDict();self.maximum_bytes=maximum_bytes;self.retained=0
        self.metrics={'hits':0,'misses':0,'formula_calls':0,'peak_retained_bytes':0}

    def get(self,key):
        entry=self.entries.get(key)
        if entry is None:self.metrics['misses']+=1;return None
        self.entries.move_to_end(key);self.metrics['hits']+=1
        return entry[0]

    def put(self,key,value):
        from .stock_matrix_reader import _resident_size
        size=_resident_size((key,value))+256
        while self.entries and self.retained+size>self.maximum_bytes:
            _,(_,old)=self.entries.popitem(last=False);self.retained-=old
        if size<=self.maximum_bytes:
            self.entries[key]=(value,size);self.retained+=size
            self.metrics['peak_retained_bytes']=max(self.metrics['peak_retained_bytes'],self.retained)


def _stable_result(value,meta,recipe):
    """Construct invariant components, clocks and revision digest once."""
    valid=meta['status']=='value' and value is not None
    require((meta['status']=='value')==valid,'financial value/status mismatch')
    require(all(c['provenance'] is None or c['provenance'].get('revision_id') and
                c['provenance'].get('usable_from') for c in meta['components']),
            'financial component revision/clock required')
    if valid and recipe=='report_announcement_ordinal':
        from datetime import date
        value=float(date.fromisoformat(str(value)[:10]).toordinal())
    elif valid and recipe=='forecast_category_code':value=float(FORECAST_CODES.get(value,0))
    else:value=float(value) if valid else None
    clocks=[c['provenance']['usable_from'] for c in meta['components'] if c['provenance'] is not None]
    clock=max(clocks,key=instant_us) if clocks else None
    # Provenance is checked while deriving, then discarded. The pinned source
    # and formula declaration permit a small diagnostic recomputation.
    summary={'status':meta['status'],'missing_reason':meta['missing_reason']}
    return value,valid,summary,clock,instant_us(clock) if clock else None


def _memoized_results(batch,event_rows,tokens,targets_by_name,outputs,field,query,unit,memo):
    from axiom_data import DataBatch,ttm,single_quarter
    records={(r['security_id'],r['report_period']):r for r in event_rows}
    provenance={(r['security_id'],r['report_period']):r for r in batch.field_meta[field]['by_key']}
    identities={(r['security_id'],r['report_period']):tokens[r['security_id']][
        tuple(str(r[k]) for k in batch.context['logical_key'])] for r in event_rows}
    results={}
    for recipe in sorted({p['derivation'] for p in outputs.values()}):
        requests=sorted({(security,period) for name,target in targets_by_name.items()
                        if outputs[name]['derivation']==recipe for security,period in target.items()})
        missing={};by={};results[recipe]=by
        for security,period in requests:
            # Owner fixes field, profile, basis, unit, stream, policy and recipe
            # version. Tokens include selected revision values/status/provenance
            # and availability; missing periods are explicit, never numeric equality.
            key=(batch.context.get('financial_selection',POLICY),recipe,security,period,tuple((p,identities.get((security,p),('MISSING',)))
                                            for p in _dependencies(period,recipe)))
            cached=memo.get(key)
            if cached is None:missing[(security,period)]=key
            else:by[(security,period)]=cached
        if not missing:continue
        if recipe in ('ttm','single_quarter'):
            changed=sorted({security for security,_ in missing});chosen=set(changed)
            periods=sorted({period for _,period in missing})
            # Reuse the original recipe with only changed securities. Its
            # output cross-product may contain extra periods; retain requests only.
            restricted=DataBatch(batch.frame.loc[batch.frame['security_id'].isin(chosen)],
                {field:{**batch.field_meta[field],'by_key':[r for r in batch.field_meta[field]['by_key'] if r['security_id'] in chosen]}},
                {**batch.context,'query':{**batch.context['query'],'symbols':changed}})
            derived=(ttm if recipe=='ttm' else single_quarter)(restricted,field=field,
                endpoint=query.filters['endpoint'],report_type=query.filters['report_type'],
                basis='cumulative_ytd',unit=unit,periods=periods,_borrow_context=True)
            memo.metrics['formula_calls']+=1
            derived_records={(r['security_id'],r['report_period']):r for r in derived.frame.to_dict('records')}
            derived_meta={(r['security_id'],r['report_period']):r for r in derived.field_meta[field]['by_key']}
        for pair,key in missing.items():
            if recipe in ('ttm','single_quarter'):
                meta=derived_meta[pair];value=derived_records[pair][field] if meta['status']=='value' else None
            elif pair in provenance:
                item=provenance[pair]
                meta={'status':item['status'],'missing_reason':item.get('missing_reason'),
                    'components':[{'security_id':pair[0],'report_period':pair[1],
                                   'status':item['status'],'provenance':item}]}
                value=records[pair].get(field)
            else:meta={'components':[],'status':'source_missing','missing_reason':'no_visible_report'};value=None
            result=_stable_result(value,meta,recipe);by[pair]=result;memo.put(key,result)
    return results


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
        require(field in manifest.get('daily_fields',{}) and spec['selection'] in (POLICY,QUARTER_POLICY,FORECAST_POLICY) and
                _projection_valid(spec) and (spec['derivation'] in ('reported','report_observation','report_announcement_ordinal','forecast_category_code') or spec['basis']=='cumulative_ytd'), 'financial materialization declaration')
        require(manifest['daily_fields'][field]=={'domain':DOMAIN,'dtype':'float64','unit':spec['unit']},
                'financial materialization typed leaf mismatch')
    if child is None: return
    financial = set(child['fields']) & set(declarations)
    if not financial: return
    require(child['context'].get('domain')==DOMAIN and child['context'].get('selection') in (POLICY,QUARTER_POLICY,FORECAST_POLICY),
            'financial evidence selection required')
    component_clocks={}
    for field in financial:
        declaration=declarations[field]
        covered=set(declaration['coverage_security_ids'])
        cutoffs={day:_cutoff(day,manifest['spec']['cutoff_by_session'][day],declaration['session_timezone']) for day in child['sessions']}
        for row in child['field_meta'][field]['by_key']:
            require(row['security_id'] in covered and
                    child['context']['snapshot_id']==declaration['event_snapshot'],'financial evidence source/coverage mismatch')
            cutoff=cutoffs[row['session']]
            require(row['query_cutoff']==cutoff and row['selection']==declaration['selection'],'financial evidence cutoff mismatch')
            if declaration['derivation'] in ('report_observation','report_announcement_ordinal','forecast_category_code'):
                periods=row['visible_report_periods'];offset=declaration['reports_back']
                require(len(periods)==len(set(periods)) and (declaration['selection']==FORECAST_POLICY or periods==sorted(periods)) and row['report_period']==(periods[-1-offset] if len(periods)>offset else None),'distinct visible report selection mismatch')
            else:
                require(row['report_period'] is None or row['report_period']==_period_back(row['current_report_period'],declaration['quarters_back']),
                        'financial lag must use economic report quarters')
            components=row['components'];key=id(components)
            if key not in component_clocks:
                clocks=[]
                for component in components:
                    provenance=component['provenance']
                    if provenance is None:continue
                    require(provenance.get('revision_id') and provenance.get('usable_from'), 'financial component revision/clock required')
                    clocks.append((instant_us(provenance['usable_from']),provenance['usable_from']))
                component_clocks[key]=max(clocks,key=lambda c:c[0]) if clocks else None
            maximum=component_clocks[key]
            require(maximum is None or maximum[0]<=instant_us(cutoff),'future financial component')
            expected=maximum[1] if maximum is not None else cutoff
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
    the latest visible eligible calendar-quarter report and an exact quarter offset.
    Observation/forecast recipes retain their distinct disclosed-report semantics.
    A bounded block shares original event partitions across all daily cutoffs
    and both current/prior-year derivations. Revision selection is Data-owned.
    Missing/retracted components remain null; original statuses/revisions are
    retained through typed missing reasons and block-level source declarations.
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
    quarterly=any('quarters_back' in p for p in outputs.values())
    require(not quarterly or all('quarters_back' in p for p in outputs.values()),'separate quarterly and observation projections required')
    if quarterly:selection=QUARTER_POLICY
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
    reason_codes=list(manifest['reason_codes'])
    if len(reason_codes)<len(REASONS):reason_codes.extend(REASONS[len(reason_codes):])
    chunks=[]; metrics={'reason_codes':reason_codes,'event_blocks':0,'event_cutoff_selections':0,'existing_field_reads':0,'added_fields':len(outputs)}
    from .feature_input_checkpoint import extension_checkpoint, atomic_json
    from .feature_input_evidence import check_input_disk, input_block_receipt
    staging,arrays=extension_checkpoint(root,source=source,fields=outputs,kinds=kinds,sessions=days,
        block_sessions=block_sessions,binding={'declarations':declarations},metrics=metrics)
    chunks=list(staging.state['chunks']);metrics=dict(staging.state['metrics'])
    reason_codes=list(metrics['reason_codes']);metrics['reason_codes']=reason_codes
    memo=_FinancialMemo()
    for metric in ('derive_seconds','evidence_seconds','encode_write_seconds','checkpoint_seconds'):
        metrics.setdefault(metric,0.0)
    empty=_stable_result(None,{'components':[],'status':'source_missing','missing_reason':'no_visible_report'},'reported')
    for start in range(staging.state['completed_dates'],len(days),block_sessions):
        check_input_disk(root,min(block_sessions,len(days)-start)*sum(a.shape[1]*a.dtype.itemsize for a in staging.arrays.values()))
        block_started=time.monotonic()
        selected=days[start:start+block_sessions]
        for si in range(0,len(query.symbols),256):
            subset=list(query.symbols[si:si+256])
            cutoffs=[_cutoff(d,spec['cutoff_by_session'][d],session_timezone) for d in selected]
            batches=iter_event_cutoffs(data,snapshot=event_snapshot,query=replace(query,symbols=tuple(subset),fields=tuple(dict.fromkeys((field,'announcement_date'))) if query.domain=='forecast_events' else query.fields),
                cutoffs=cutoffs,maximum_cache_bytes=maximum_event_cache_bytes,owner=event_owner,_borrow_context=True)
            metrics['event_blocks']+=1
            for offset,(day,batch) in enumerate(zip(selected,batches)):
                metrics['event_cutoff_selections']+=1
                require(batch.context['snapshot_id']==event_snapshot and
                        instant_us(batch.context['query']['cutoff'])==instant_us(cutoffs[offset]),'financial selected batch binding')
                derive_started=time.monotonic()
                event_rows=event_owner.selected_records
                selected_tokens=event_owner.selected_tokens
                if quarterly:
                    batch,event_rows,selected_tokens=_quarter_projection(batch,event_rows,selected_tokens,day)
                require(all(row['security_id'] in subset and
                    ('endpoint' not in query.filters or row.get('endpoint')==query.filters['endpoint']) and
                    ('report_type' not in query.filters or row.get('report_type')==query.filters['report_type'])
                    for row in event_rows),'unexpected financial event stream')
                targets_by_name=_projection_targets(event_rows,outputs,subset,day,forecast=query.domain=='forecast_events')
                derived_by_recipe=_memoized_results(batch,event_rows,selected_tokens,
                    targets_by_name,outputs,field,query,unit,memo)
                metrics['derive_seconds']+=time.monotonic()-derive_started
                evidence_started=time.monotonic()
                positions=[source.security_positions[security] for security in subset]
                for name,projection in outputs.items():
                    targets=targets_by_name[name]
                    by=derived_by_recipe[projection['derivation']]
                    payload={kind:[] for kind in kinds}
                    for security in subset:
                        period=targets.get(security)
                        value,valid,meta,component_clock,component_us=by.get((security,period),empty)
                        clock=component_clock or cutoffs[offset]
                        require(component_us is None or component_us<=instant_us(cutoffs[offset]),
                                'future financial component')
                        require(not valid or np.isfinite(value),'nonfinite financial input')
                        reason=None if valid else meta['missing_reason']
                        require(reason is None or isinstance(reason,str) and bool(reason),'invalid financial missing state')
                        if reason not in reason_codes:reason_codes.append(reason)
                        require(len(reason_codes)<=256,'financial reason dictionary exceeds uint8')
                        for kind,v in {'values':value if valid else 0,'original_values':value if valid else 0,
                            'validity':valid,'available':component_us if component_us is not None else instant_us(clock),'available_validity':True,
                            'reasons':reason_codes.index(reason)}.items():payload[kind].append(v)
                    for kind,values in payload.items():arrays[name+'::'+kind][start+offset,positions]=values
                metrics['evidence_seconds']+=time.monotonic()-evidence_started
        encode_started=time.monotonic()
        child=input_block_receipt(spec=spec,sessions=selected,fields=outputs,
            context={'domain':DOMAIN,'selection':selection,'snapshot_id':event_snapshot},
            definitions={name:{'dtype':'float64','unit':unit} for name in outputs},
            bindings={name:digest({k:v for k,v in declaration.items() if k not in
                ('report_start','report_end','coverage_security_ids')}) for name,declaration in declarations.items()})
        chunks.append({'sessions':selected,'evidence':[write_part(root,child,'evidence_ref')]})
        metrics['encode_write_seconds']+=time.monotonic()-encode_started
        metrics['event_reader']=dict(event_owner.metrics);metrics['financial_memo']=dict(memo.metrics)
        metrics['saved_cell_proof_rows']=0
        metrics['block_receipt_bytes']=metrics.get('block_receipt_bytes',0)+Path(chunks[-1]['evidence'][0]['path']).stat().st_size
        checkpoint_started=time.monotonic()
        staging.commit(start+len(selected),chunks,metrics)
        metrics['checkpoint_seconds']+=time.monotonic()-checkpoint_started
        metrics['last_block_seconds']=time.monotonic()-block_started
        if progress is not None: progress({'completed_dates':start+len(selected),'total_dates':len(days),**metrics})
    for name in outputs:
        for kind,(_,dtype) in kinds.items():
            value=arrays[name+'::'+kind]
            manifest['buffers'][name+'::'+kind],arrays[name+'::'+kind]=seal_feature_buffer(root,value,dtype=dtype,shape=shape,retain_staging=True)
    manifest['reason_codes']=reason_codes
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


def extend_financial_feature_sources(data, *, source, bundles, destination,
        event_snapshot, report_start, report_end, maximum_added_bytes,
        maximum_event_cache_bytes, maximum_group_bytes, progress=None,
        stock_block_limit=None):
    """One stock-block traversal for all required compatible event inputs.

    Published daily ABI and original Data recipes remain unchanged. Only the
    final complete source is visible; the existing checkpoint commits securities.
    """
    import math
    import numpy as np
    from axiom_data import DataBatch,EventQuery,EventReadOwner
    from axiom_data.financial import FINANCIAL_RECIPE_VERSION
    from .stock_feature_columns import FeatureColumnSource,REASONS
    from .feature_input_buffers import seal_feature_buffer
    from .feature_input_checkpoint import FeatureInputCheckpoint,atomic_json
    from .feature_input_evidence import input_block_receipt,check_input_disk
    manifest=deepcopy(source.manifest);spec=manifest['spec'];days=spec['read_sessions'];symbols=spec['universe']
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    require(spec['pit_policy']=='best_effort_vendor_v1','supported financial PIT policy required')
    streams={};declarations={};field_bundles={}
    for name,bundle in bundles.items():
        outputs={n:p for n,p in bundle['outputs'].items() if n not in source.physical_fields()}
        if not outputs:continue
        quarterly=any('quarters_back' in p for p in outputs.values())
        require(not quarterly or all('quarters_back' in p for p in outputs.values()),'separate quarterly/observation recipes')
        require(all(_projection_valid(p) for p in outputs.values()),'explicit financial projections required')
        selection=QUARTER_POLICY if quarterly else FORECAST_POLICY if bundle['domain']=='forecast_events' else POLICY
        if quarterly:require(bundle.get('selection')==selection,'quarter declaration differs from producer rule')
        filters={k:bundle[k] for k in ('endpoint','report_type') if bundle[k] is not None}
        key=(event_snapshot,bundle['domain'],tuple(sorted(filters.items())),report_start,report_end,
             spec['pit_policy'],bundle['session_timezone'],quarterly)
        stream=streams.setdefault(key,{'domain':bundle['domain'],'filters':filters,'fields':[],
            'bundles':[],'zone':bundle['session_timezone'],'quarterly':quarterly})
        fields=[bundle['field']]+(['announcement_date'] if bundle['domain']=='forecast_events' else [])
        stream['fields']=list(dict.fromkeys(stream['fields']+fields))
        stream['bundles'].append(name);field_bundles[name]={**bundle,'outputs':outputs,'selection':selection}
        for output,projection in outputs.items():
            require(output not in declarations,'duplicate financial output')
            declarations[output]={**projection,'source_domain':bundle['domain'],'source_field':bundle['field'],
                'endpoint':bundle['endpoint'],'report_type':bundle['report_type'],'basis':bundle['basis'],
                'unit':bundle['unit'],'selection':selection,'session_timezone':bundle['session_timezone'],
                'recipe_version':FINANCIAL_RECIPE_VERSION,'report_start':report_start,'report_end':report_end,
                'event_snapshot':event_snapshot,'coverage_security_ids':list(symbols)}
    require(declarations,'missing event outputs required')
    require(len(days)*len(symbols)*27*len(declarations)<=maximum_added_bytes,'financial output buffer budget exceeded')
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
        'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
        'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    layout={name+'::'+kind:{'dtype':np.dtype(dtype).str,'shape':[len(days),len(symbols)]}
            for name in declarations for kind,(dtype,_) in kinds.items()}
    reasons=list(manifest['reason_codes'])
    if len(reasons)<len(REASONS):reasons.extend(REASONS[len(reasons):])
    metrics={'reason_codes':reasons,'completed_securities':0,'total_securities':len(symbols),
        'total_dates':len(days),'added_fields':len(declarations),'compatible_streams':len(streams),
        'derive_seconds':0.0,'fill_seconds':0.0,'encode_write_seconds':0.0,'checkpoint_seconds':0.0,
        'derived_spans':0,'broadcast_cells':0,'saved_cell_proof_rows':0,'stock_blocks':[],
        'memo_hits':0,'memo_misses':0,'formula_calls':0}
    staging=FeatureInputCheckpoint(root,binding={'source_ref':source.identity,'declarations':declarations,
        'execution':'stock_block_event_spans_v1'},layout=layout,sessions=symbols,metrics=metrics,axis=1)
    arrays={**source.arrays,**staging.arrays};metrics=staging.state['metrics'];reasons=metrics['reason_codes']
    started=time.monotonic();last_report=started
    with EventReadOwner(data,snapshot=event_snapshot,maximum_cache_bytes=maximum_event_cache_bytes,
                        maximum_partition_rows=max(1,maximum_event_cache_bytes//64)) as owner:
        # Estimate stock width from the largest compatible projected stream,
        # not fixed source/field counts. Actual decoded bytes are checked by Data.
        largest=1
        for stream in streams.values():
            native=owner.manifest['domains'][stream['domain']]
            rows=sum(p['rows'] for p in native['partitions'] if
                not p.get('partition','').startswith('period-') or report_start[:4]<=p['partition'][7:]<=report_end[:4])
            width=len(stream['fields'])+len(native['contract']['logical_key'])+12
            largest=max(largest,math.ceil(rows/max(1,len(symbols)))*width*256)
            for name in stream['bundles']:
                bundle=field_bundles[name];actual=native['contract']['fields'][bundle['field']]
                require(actual.get('unit')==bundle['unit'] or
                    actual.get('dtype')=='string' and bundle['unit']=='dimensionless' or
                    actual.get('dtype')=='date' and bundle['unit']=='days','financial source unit mismatch')
                require(all(p['derivation'] not in ('ttm','single_quarter') for p in bundle['outputs'].values()) or
                    actual.get('basis')=='cumulative_ytd','financial source basis mismatch')
        width=max(1,min(len(symbols),maximum_group_bytes//largest))
        if stock_block_limit is not None:
            require(type(stock_block_limit) is int and stock_block_limit>0,'positive stock block limit')
            width=min(width,stock_block_limit)
        metrics['planned_stock_block_size']=width;metrics['estimated_group_bytes_per_security']=largest
        first=staging.state['completed_dates']
        while first<len(symbols):
            settings=root/'resource-settings.json'
            if settings.exists():
                import json
                requested=json.loads(settings.read_text())['stock_block_size']
                require(type(requested) is int and 0<requested<=len(symbols),'valid stock resource setting required')
                if requested!=width:
                    metrics.setdefault('resource_adjustments',[]).append({'completed_securities':first,'before':width,'after':requested,
                        'group_budget_bytes':maximum_group_bytes,'basis':'explicit resource-settings.json; actual decoded budget remains enforced'})
                    width=requested
            stop=min(first+width,len(symbols));subset=symbols[first:stop];block_started=time.monotonic()
            check_input_disk(root)
            for stream in streams.values():
                cutoffs=[_cutoff(d,spec['cutoff_by_session'][d],stream['zone']) for d in days]
                cutoff_us=np.asarray([instant_us(c) for c in cutoffs],dtype='<i8')
                query=EventQuery(stream['domain'],tuple(stream['fields']),tuple(subset),report_start,report_end,
                    cutoffs[-1],spec['pit_policy'],'report_period',stream['filters'])
                memos={name:_FinancialMemo() for name in stream['bundles']}
                positions={s:first+i for i,s in enumerate(subset)}
                for security,start,end,batch in owner.iter_security_spans(query=query,cutoffs=cutoffs,sessions=days,
                        economic_boundary=stream['domain']!='forecast_events',maximum_group_bytes=maximum_group_bytes):
                    rows=owner.selected_records;tokens=owner.selected_tokens
                    if stream['quarterly']:batch,rows,tokens=_quarter_projection(batch,rows,tokens,days[start])
                    column=positions[security]
                    for name in stream['bundles']:
                        bundle=field_bundles[name];field=bundle['field'];outputs=bundle['outputs']
                        # Preserve field-specific metadata/clocks; never replace
                        # independent source availability with a common row clock.
                        selected=DataBatch(batch.frame,{field:batch.field_meta[field]},
                            {**batch.context,'query':{**batch.context['query'],'fields':[field]}})
                        single=replace(query,fields=(field,),symbols=(security,))
                        targets=_projection_targets(rows,outputs,[security],days[start],forecast=stream['domain']=='forecast_events')
                        tick=time.monotonic()
                        values=_memoized_results(selected,rows,tokens,targets,outputs,field,single,bundle['unit'],memos[name])
                        metrics['derive_seconds']+=time.monotonic()-tick;metrics['derived_spans']+=1
                        tick=time.monotonic()
                        for output,projection in outputs.items():
                            period=targets[output].get(security)
                            result=values[projection['derivation']].get((security,period))
                            if result is None:value,valid,meta,clock,clock_us=None,False,{'missing_reason':'no_visible_report'},None,None
                            else:value,valid,meta,clock,clock_us=result
                            require(clock_us is None or clock_us<=cutoff_us[start],'future financial component')
                            require(not valid or np.isfinite(value),'nonfinite financial input')
                            reason=None if valid else meta['missing_reason']
                            require(reason is None or isinstance(reason,str) and bool(reason),'financial missing state required')
                            if reason not in reasons:reasons.append(reason)
                            require(len(reasons)<=256,'financial reason dictionary exceeds uint8')
                            target=(slice(start,end),column)
                            for kind in ('values','original_values'):arrays[output+'::'+kind][target]=value if valid else 0
                            arrays[output+'::validity'][target]=valid
                            arrays[output+'::reasons'][target]=reasons.index(reason)
                            arrays[output+'::available'][target]=clock_us if clock_us is not None else cutoff_us[start:end]
                            arrays[output+'::available_validity'][target]=True
                            metrics['broadcast_cells']+=end-start
                        metrics['fill_seconds']+=time.monotonic()-tick
                    if progress is not None and time.monotonic()-last_report>=10:
                        progress({**metrics,'active_security':security,'active_domain':stream['domain'],
                                  'event_reader':dict(owner.metrics),'elapsed_seconds':time.monotonic()-started})
                        last_report=time.monotonic()
                for memo in memos.values():
                    metrics['memo_hits']+=memo.metrics['hits'];metrics['memo_misses']+=memo.metrics['misses']
                    metrics['formula_calls']+=memo.metrics['formula_calls']
                del memos
            metrics['completed_securities']=stop;metrics['event_reader']=dict(owner.metrics)
            metrics['stock_blocks'].append({'start':first,'stop':stop,'seconds':time.monotonic()-block_started})
            tick=time.monotonic();staging.commit(stop,[],metrics);metrics['checkpoint_seconds']+=time.monotonic()-tick
            if progress is not None:progress(dict(metrics))
            first=stop
        chunks=[];tick=time.monotonic()
        for offset in range(0,len(days),32):
            child=input_block_receipt(spec=spec,sessions=days[offset:offset+32],fields=declarations,
                context={'domain':DOMAIN,'snapshot_id':event_snapshot},
                definitions={n:{'dtype':'float64','unit':d['unit']} for n,d in declarations.items()},
                bindings={n:digest({k:v for k,v in d.items() if k not in ('report_start','report_end','coverage_security_ids')}) for n,d in declarations.items()})
            chunks.append({'sessions':days[offset:offset+32],'evidence':[write_part(root,child,'evidence_ref')]})
        metrics['block_receipt_bytes']=sum(Path(c['evidence'][0]['path']).stat().st_size for c in chunks)
        metrics['encode_write_seconds']+=time.monotonic()-tick
        for name in declarations:
            for kind,(_,dtype) in kinds.items():
                key=name+'::'+kind
                manifest['buffers'][key],arrays[key]=seal_feature_buffer(root,arrays[key],dtype=dtype,
                    shape=(len(days),len(symbols)),retain_staging=True)
        manifest['reason_codes']=reasons
        manifest.setdefault('daily_fields',{}).update({n:{'domain':DOMAIN,'dtype':'float64','unit':d['unit']} for n,d in declarations.items()})
        manifest.setdefault('financial_materializations',{}).update(declarations)
        manifest.setdefault('daily_extensions',[]).append({'fields':list(declarations),'chunks':chunks})
        previous=manifest['source_profiles'].get(DOMAIN)
        if previous is None:
            first=next(iter(declarations.values()));native=owner.manifest['domains'][first['source_domain']]
            previous={'contract':{'contract_id':'research.financial_materialization.v1','logical_key':['security_id','session'],'fields':{}},
                'source_profile':{'id':'research.financial_materialization.v1','financial_inputs':{k:native[k] for k in ('contract','source_profile')},
                    'financial_inputs_by_field':{},'projections':{},'not_native_daily':True}}
        profile=deepcopy(previous);by_field=profile['source_profile'].setdefault('financial_inputs_by_field',{
            n:profile['source_profile']['financial_inputs'] for n in profile['source_profile']['projections']})
        for name,declaration in declarations.items():
            native=owner.manifest['domains'][declaration['source_domain']]
            profile['contract']['fields'][name]={'dtype':'float64','unit':declaration['unit']}
            by_field[name]={k:native[k] for k in ('contract','source_profile')}
            profile['source_profile']['projections'][name]={k:v for k,v in declaration.items() if k not in
                ('event_snapshot','report_start','report_end','coverage_security_ids')}
        manifest['source_profiles'][DOMAIN]=profile
        validate_financial_materializations(manifest)
        staging.commit(len(symbols),chunks,metrics)
        manifest.pop('source_ref');manifest=seal(manifest,'source_ref');atomic_json(root/'source.json',manifest)
        staging.retire(manifest)
    return FeatureColumnSource(manifest,arrays),metrics

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
        require(field in manifest.get('daily_fields',{}) and spec['selection']==POLICY and
                spec['derivation'] in ('ttm','single_quarter','reported') and type(spec['quarters_back']) is int and
                spec['quarters_back']>=0 and (spec['derivation']=='reported' or spec['basis']=='cumulative_ytd'), 'financial materialization declaration')
        require(manifest['daily_fields'][field]=={'domain':DOMAIN,'dtype':'float64','unit':spec['unit']},
                'financial materialization typed leaf mismatch')
    if child is None: return
    financial = set(child['fields']) & set(declarations)
    if not financial: return
    require(child['context'].get('domain')==DOMAIN and child['context'].get('selection')==POLICY,
            'financial evidence selection required')
    for field in financial:
        declaration=declarations[field]
        for row in child['field_meta'][field]['by_key']:
            require(row['security_id'] in declaration['coverage_security_ids'] and
                    child['context']['snapshot_id']==declaration['event_snapshot'],'financial evidence source/coverage mismatch')
            cutoff=_cutoff(row['session'],manifest['spec']['cutoff_by_session'][row['session']],declaration['session_timezone'])
            require(row['query_cutoff']==cutoff and row['selection']==POLICY,'financial evidence cutoff mismatch')
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
        maximum_added_bytes=64*1024**2, maximum_event_cache_bytes=64*1024**2):
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
    require(isinstance(query,EventQuery) and query.domain in ('financial_events','cash_flow_events','balance_sheet_events','financial_indicator_events') and
            query.time_field=='report_period' and query.fields==(field,) and query.purpose=='decision_facts',
            'one explicit financial event field required')
    manifest=deepcopy(source.manifest); spec=manifest['spec']; days=spec['read_sessions']; symbols=spec['universe']
    require(bool(query.symbols) and list(query.symbols)==sorted(set(query.symbols)) and
            set(query.symbols)<=set(symbols) and query.pit_policy==spec['pit_policy']=='best_effort_vendor_v1','financial source scope/policy mismatch')
    require(query.filters.get('endpoint'),'financial stream endpoint required')
    require(type(block_sessions) is int and 0<block_sessions<=32,'one to 32 financial sessions per block')
    require(type(outputs) is dict and outputs and not set(outputs)&set(source.physical_fields()),'new financial outputs required')
    for name,projection in outputs.items():
        require(name.isidentifier() and set(projection)=={'derivation','quarters_back'} and
                projection['derivation'] in ('ttm','single_quarter','reported') and type(projection['quarters_back']) is int and
                projection['quarters_back']>=0,'explicit financial output recipe required')
    require(event_snapshot not in ('','current','latest'),'pinned independent financial Snapshot required')
    snapshot=data.describe_snapshot(snapshot=event_snapshot,domains=[query.domain]); native=snapshot['domains'][query.domain]
    actual=native['contract']['fields'][field]
    require(actual.get('unit')==unit and (all(p['derivation']=='reported' for p in outputs.values()) or actual.get('basis')=='cumulative_ytd'),'financial input unit/basis mismatch')
    shape=(len(days),len(symbols)); required=len(days)*len(symbols)*27*len(outputs)
    require(type(maximum_added_bytes) is int and required<=maximum_added_bytes,'financial output buffer budget exceeded')
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    arrays=dict(source.arrays)
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
           'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
           'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    for name in outputs:
        for kind,(dtype,_) in kinds.items(): arrays[name+'::'+kind]=np.zeros(shape,dtype=dtype)
    declarations={name:{**projection,'source_domain':query.domain,'source_field':field,
        'endpoint':query.filters['endpoint'],'report_type':query.filters.get('report_type'),'basis':actual.get('basis','reported'),
        'unit':unit,'selection':POLICY,'session_timezone':session_timezone,'recipe_version':FINANCIAL_RECIPE_VERSION,
        'report_start':query.start,'report_end':query.end,'event_snapshot':event_snapshot,
        'coverage_security_ids':list(query.symbols)} for name,projection in outputs.items()}
    chunks=[]; metrics={'event_blocks':0,'event_cutoff_selections':0,'existing_field_reads':0,'added_fields':len(outputs)}
    for start in range(0,len(days),block_sessions):
        selected=days[start:start+block_sessions]; metadata={n:{'dtype':'float64','unit':unit,'by_key':[]} for n in outputs}
        contexts=[]
        for si in range(0,len(query.symbols),256):
            subset=list(query.symbols[si:si+256])
            cutoffs=[_cutoff(d,spec['cutoff_by_session'][d],session_timezone) for d in selected]
            batches=iter_event_cutoffs(data,snapshot=event_snapshot,query=replace(query,symbols=tuple(subset)),
                cutoffs=cutoffs,maximum_cache_bytes=maximum_event_cache_bytes)
            metrics['event_blocks']+=1
            for offset,(day,batch) in enumerate(zip(selected,batches)):
                metrics['event_cutoff_selections']+=1
                require(batch.context['snapshot_id']==event_snapshot and
                        instant_us(batch.context['query']['cutoff'])==instant_us(cutoffs[offset]),'financial selected batch binding')
                contexts.append(deepcopy(batch.context))
                latest={}
                for row in batch.frame.to_dict('records'):
                    require(row['security_id'] in subset and row['endpoint']==query.filters['endpoint'] and
                            ('report_type' not in query.filters or row.get('report_type')==query.filters['report_type']),'unexpected financial event stream')
                    if row['report_period']<=day:
                        latest[row['security_id']]=max(latest.get(row['security_id'],row['report_period']),row['report_period'])
                # Prepare each Data derivation once for the union of current
                # and prior-year target periods, not once per output leaf.
                targets_by_name={name:{s:_period_back(p,projection['quarters_back']) for s,p in latest.items()}
                                 for name,projection in outputs.items()}
                derived_by_recipe={}
                for recipe in {p['derivation'] for p in outputs.values()}:
                    periods=sorted({p for name,target in targets_by_name.items()
                                    if outputs[name]['derivation']==recipe for p in target.values()})
                    if recipe=='reported':
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
                        value=float(value) if valid else None
                        clocks=[c['provenance']['usable_from'] for c in meta['components'] if c['provenance'] is not None]
                        clock=max(clocks,key=instant_us) if clocks else cutoffs[offset]
                        bases=sorted({c['provenance']['availability_basis'] for c in meta['components'] if c['provenance'] is not None})
                        evidence={'security_id':security,'session':day,'report_period':period,'current_report_period':latest.get(security),
                            'selection':POLICY,'query_cutoff':cutoffs[offset],'status':meta['status'],
                            'financial_missing_reason':meta['missing_reason'],'missing_reason':None if valid else 'source_missing',
                            'usable_from':clock,'availability_basis':bases[0] if len(bases)==1 else 'derived_component_max' if bases else 'no_visible_report_at_query_cutoff',
                            'component_availability_bases':bases,
                            'components':meta['components'],'selected_value':value,'revision_id':digest(meta)}
                        metadata[name]['by_key'].append(evidence)
                        position=(start+offset,source.security_positions[security])
                        for kind,v in {'values':value if valid else 0,'original_values':value if valid else 0,
                            'validity':valid,'available':instant_us(clock),'available_validity':True,
                            'reasons':0 if valid else REASONS.index('source_missing')}.items():arrays[name+'::'+kind][position]=v
        child=seal({'contract_version':'reader_column_evidence_v1','context':{'domain':DOMAIN,'selection':POLICY,
            'snapshot_id':event_snapshot,'event_contexts':contexts},'field_meta':metadata,'fields':list(outputs),'sessions':selected},'evidence_ref')
        check_manifest={**manifest,'daily_fields':{**manifest.get('daily_fields',{}),**{n:{'domain':DOMAIN,'dtype':'float64','unit':unit} for n in outputs}},
                        'financial_materializations':{**manifest.get('financial_materializations',{}),**declarations}}
        validate_financial_materializations(check_manifest,child)
        chunks.append({'sessions':selected,'evidence':[write_part(root,child,'evidence_ref')]})
    for name in outputs:
        for kind,(_,dtype) in kinds.items():
            value=arrays[name+'::'+kind]
            manifest['buffers'][name+'::'+kind]=write_buffer(root,value.astype('u1').reshape(-1) if dtype=='bool_u8' else value.reshape(-1),dtype=dtype,shape=list(shape))
    manifest.setdefault('daily_fields',{}).update({n:{'domain':DOMAIN,'dtype':'float64','unit':unit} for n in outputs})
    manifest.setdefault('financial_materializations',{}).update(declarations)
    manifest.setdefault('daily_extensions',[]).append({'fields':list(outputs),'chunks':chunks})
    profile={'contract':{'contract_id':'research.financial_materialization.v1','logical_key':['security_id','session'],
                        'fields':{n:{'dtype':'float64','unit':unit} for n in outputs}},
             'source_profile':{'id':'research.financial_materialization.v1','financial_inputs':{k:native[k] for k in ('contract','source_profile')},
                'projections':{n:{k:v for k,v in d.items() if k not in ('event_snapshot','report_start','report_end','coverage_security_ids')} for n,d in declarations.items()},'selection':POLICY,'not_native_daily':True}}
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
    manifest.pop('source_ref');manifest=seal(manifest,'source_ref');write_json(root/'source.json',manifest)
    return FeatureColumnSource(manifest,arrays),metrics

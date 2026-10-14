"""SW L1 interval projection; terminal supplier history, explicitly not strict PIT."""
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path

from .stock_fold_inputs import require, seal
from .stock_artifacts import write_json, digest
from .stock_matrix_storage import instant_us, write_part
from .stock_feature_columns import FeatureColumnSource, REASONS
from .feature_input_buffers import allocate_feature_buffer, seal_feature_buffer


def extend_industry_feature_source(data, *, source, snapshot, destination,
                                   block_sessions=5, maximum_added_bytes=512*1024**2, progress=None):
    """Bind L1 by [in_date,out_date); missing/conflicting dates stay unknown.

    Dates are supplier effective-date assumptions, NOT publication timestamps.
    Actual receipt/revision evidence is retained. Classification is not filled
    before the earliest dated interval. Simultaneous distinct L1 codes conflict.
    Same L1 across different L2/L3 rows is not a first-level conflict. No source
    collection occurs. An immutable added input also binds CS reference identity.
    """
    field='industry_code';domain='materialized_industry_intervals'
    require(field not in source.physical_fields(),'industry input already exists')
    manifest=deepcopy(source.manifest);spec=manifest['spec'];days=spec['read_sessions'];symbols=spec['universe']
    require(0<block_sessions<=32,'industry block size must be one to 32 sessions')
    shape=(len(days),len(symbols));require(len(days)*len(symbols)*27<=maximum_added_bytes,'industry buffer budget exceeded')
    native=data.industry_intervals(snapshot=snapshot,security_ids=symbols)
    require(native['qualification']=='sw_supplier_intervals_best_effort_v1' and native['publication_pit'] is False,'explicit industry best effort required')
    intervals=defaultdict(list);invalid=Counter()
    for original in native['records']:
        row={k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in original.items()}
        begin=row.get('effective_from');end=row.get('effective_to');code=row.get('l1_code')
        if not begin or not code or not code.endswith('.SI') or not code[:-3].isdigit() or end and end<begin:
            invalid['invalid_interval']+=1;continue
        intervals[row['security_id']].append((begin,end,code,row))
    root=Path(destination);root.mkdir(parents=True,exist_ok=True);arrays=dict(source.arrays)
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
           'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
           'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    profile={'id':'research.sw_l1_intervals.v1','policy':native['qualification'],
             'boundary':'in_date inclusive, out_date exclusive; supplier-boundary assumption',
             'taxonomy':'supplier-returned L1; historical taxonomy vintage not certified',
             'clock':'effective_from local midnight assumption; NOT publication PIT',
             'snapshot_id':snapshot,'native':native['domain'],'unknown':'missing or conflicting dated L1 interval',
             'numeric_code':'digits of native L1 .SI code; categorical, never ordinal'}
    from .feature_input_checkpoint import extension_checkpoint, atomic_json
    from .feature_input_evidence import check_input_disk
    staging,arrays=extension_checkpoint(root,source=source,fields=[field],kinds=kinds,sessions=days,
        block_sessions=block_sessions,binding={'snapshot':snapshot,'profile':profile},metrics={'coverage':{}})
    chunks=list(staging.state['chunks']);counts=Counter(staging.state['metrics']['coverage'])
    for start in range(staging.state['completed_dates'],len(days),block_sessions):
        check_input_disk(root,min(block_sessions,len(days)-start)*sum(a.shape[1]*a.dtype.itemsize for a in staging.arrays.values()))
        selected=days[start:start+block_sessions];evidence=[]
        for offset,day in enumerate(selected):
            for j,security in enumerate(symbols):
                candidates=[r for begin,end,code,r in intervals[security] if begin<=day and (not end or day<end)]
                codes={r['l1_code'] for r in candidates};valid=len(codes)==1
                reason=None if valid else 'source_missing'
                detail='value' if valid else 'conflicting_intervals' if codes else 'no_dated_interval'
                counts[detail]+=1
                value=float(next(iter(codes))[:-3]) if valid else 0.
                clock=max((instant_us(r['effective_from']+'T00:00:00+08:00') for r in candidates),default=0)
                require(not clock or clock<=instant_us(spec['cutoff_by_session'][day]),'future industry interval')
                payload={'values':value,'original_values':value,'validity':valid,'available':clock,'available_validity':bool(clock),'reasons':REASONS.index(reason)}
                for kind,v in payload.items():arrays[field+'::'+kind][start+offset,j]=v
                evidence.append({'security_id':security,'session':day,'missing_reason':reason,
                    'interval_status':detail,'selected_intervals':candidates,
                    'usable_from':max((r['effective_from'] for r in candidates),default=day)+'T00:00:00+08:00',
                    'qualification':native['qualification']})
        child=seal({'contract_version':'reader_column_evidence_v1',
            'context':{'domain':domain,'snapshot_id':snapshot,'projection':profile},
            'field_meta':{field:{'dtype':'float64','unit':'dimensionless','by_key':evidence}},'fields':[field],'sessions':selected},'evidence_ref')
        chunks.append({'sessions':selected,'evidence':[write_part(root,child,'evidence_ref')]})
        staging.commit(start+len(selected),chunks,{'coverage':dict(counts)})
        if progress is not None: progress({'completed_dates':start+len(selected),'total_dates':len(days),'coverage':dict(counts)})
    for kind,(_,dtype) in kinds.items():
        manifest['buffers'][field+'::'+kind],arrays[field+'::'+kind]=seal_feature_buffer(root,arrays[field+'::'+kind],dtype=dtype,shape=shape,retain_staging=True)
    manifest.setdefault('daily_fields',{})[field]={'domain':domain,'dtype':'float64','unit':'dimensionless'}
    manifest['source_profiles'][domain]={'contract':{'contract_id':'research.sw_l1_intervals.v1','logical_key':['security_id','session'],
        'fields':{field:{'dtype':'float64','unit':'dimensionless'}}},'source_profile':profile}
    manifest.setdefault('daily_extensions',[]).append({'fields':[field],'chunks':chunks})
    manifest.pop('source_ref');manifest=seal(manifest,'source_ref');atomic_json(root/'source.json',manifest)
    return FeatureColumnSource(manifest,arrays),{'added_fields':1,'interval_rows':len(native['records']),'coverage':dict(counts),'invalid_intervals':dict(invalid),'publication_pit':False}

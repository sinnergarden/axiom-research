"""Explicit benchmark identity and session-aligned input broadcast."""
from copy import deepcopy
from pathlib import Path

from .stock_fold_inputs import require, seal
from .stock_artifacts import write_json
from .stock_matrix_storage import instant_us, write_buffer, write_part
from .stock_feature_columns import FeatureColumnSource, REASONS


def extend_benchmark_feature_source(data, *, source, snapshot, benchmark_security_id,
                                    destination, block_sessions=20,
                                    maximum_added_bytes=512*1024**2):
    """Read one benchmark observation per session, then broadcast to stock rows.

    No implicit index, stock-to-index alias, forward fill, or price adjustment.
    Every receipt keeps the native benchmark identity and selected revision.
    """
    import numpy as np
    from axiom_data import QuerySpec
    field='benchmark_close';domain='materialized_benchmark_daily'
    require(field not in source.physical_fields(),'benchmark input already exists')
    require(isinstance(snapshot,str) and snapshot not in ('','current','latest') and isinstance(benchmark_security_id,str) and bool(benchmark_security_id),'explicit frozen benchmark required')
    manifest=deepcopy(source.manifest);spec=manifest['spec'];days=spec['read_sessions'];symbols=spec['universe']
    require(0<block_sessions<=32,'benchmark block size must be one to 32 sessions')
    shape=(len(days),len(symbols))
    require(len(days)*len(symbols)*27<=maximum_added_bytes,'benchmark buffer budget exceeded')
    native=data.describe_snapshot(snapshot=snapshot,domains=['benchmark_daily'])['domains']['benchmark_daily']
    physical=native['contract']['fields']['close']
    require(physical['dtype']=='float64' and physical['unit']=='index points','canonical benchmark close required')
    from .feature_input_buffers import allocate_feature_buffer, seal_feature_buffer
    arrays=dict(source.arrays);root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
           'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
           'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    for kind,(dtype,_) in kinds.items():arrays[field+'::'+kind]=allocate_feature_buffer(root,dtype=dtype,shape=shape)
    chunks=[];calls=0
    for start in range(0,len(days),block_sessions):
        selected=days[start:start+block_sessions]
        query=QuerySpec('benchmark_daily',('close',),(benchmark_security_id,),tuple(selected),
                        spec['pit_policy'],{d:spec['cutoff_by_session'][d] for d in selected})
        batch=data.read(snapshot=snapshot,query=query);calls+=1
        records={r['session']:r for r in batch.frame.to_dict('records')}
        metas={r['session']:r for r in batch.field_meta['close']['by_key']}
        require(len(records)==len(selected) and set(records)==set(metas)==set(selected),'one benchmark row per session required')
        evidence=[]
        for i,day in enumerate(selected):
            row=records[day];meta=metas[day]
            require(row['security_id']==meta['security_id']==benchmark_security_id,'benchmark identity mismatch')
            valid=meta.get('missing_reason') is None
            value=float(row['close']) if valid else 0.0
            require(not valid or np.isfinite(value),'nonfinite benchmark price')
            clock=instant_us(meta['usable_from']) if meta.get('usable_from') else 0
            require(not clock or clock<=instant_us(spec['cutoff_by_session'][day]),'future benchmark input')
            payload={'values':value,'original_values':value,'validity':valid,'available':clock,
                     'available_validity':bool(clock),'reasons':REASONS.index(meta.get('missing_reason'))}
            for kind,value in payload.items():arrays[field+'::'+kind][start+i,:]=value
            evidence.extend({**deepcopy(meta),'security_id':s,'benchmark_security_id':benchmark_security_id} for s in symbols)
        child=seal({'contract_version':'reader_column_evidence_v1',
            'context':{'domain':domain,'snapshot_id':snapshot,'benchmark_security_id':benchmark_security_id,
                       'native_context':batch.context,'projection':'same_session_broadcast_v1'},
            'field_meta':{field:{'dtype':'float64','unit':'index points','by_key':evidence}},
            'fields':[field],'sessions':selected},'evidence_ref')
        chunks.append({'sessions':selected,'evidence':[write_part(root,child,'evidence_ref')]})
    for kind,(_,dtype) in kinds.items():
        value=arrays[field+'::'+kind]
        manifest['buffers'][field+'::'+kind],arrays[field+'::'+kind]=seal_feature_buffer(root,value,dtype=dtype,shape=shape)
    manifest.setdefault('daily_fields',{})[field]={'domain':domain,'dtype':'float64','unit':'index points'}
    manifest['source_profiles'][domain]={'contract':{'contract_id':'research.benchmark_broadcast.v1',
        'logical_key':['security_id','session'],'fields':{field:{'dtype':'float64','unit':'index points'}}},
        'source_profile':{'id':'research.benchmark_broadcast.v1','benchmark_security_id':benchmark_security_id,
            'projection':'same_session_broadcast_v1','native_contract':native['contract'],'native_profile':native['source_profile']}}
    manifest.setdefault('daily_extensions',[]).append({'fields':[field],'chunks':chunks})
    manifest.pop('source_ref');manifest=seal(manifest,'source_ref');write_json(root/'source.json',manifest)
    return FeatureColumnSource(manifest,arrays),{'data_read_calls':calls,'added_fields':1,'existing_field_reads':0}

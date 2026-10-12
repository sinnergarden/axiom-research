"""One admitted physical price source for the bounded Core column handoff.

This adapter selects through Data and reuses Data's adjustment array kernel.
It performs no Feature mathematics. Only the declared vendor policy currently
has a proved cutoff-invariant daily selection; stricter policies are rejected.
"""
from dataclasses import replace
from pathlib import Path

from .stock_artifacts import digest, file_digest, write_json, _verify_ref
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import instant_us, write_buffer, write_part
from .stock_matrix_feature_producer import FIELDS, prepare_matrix_qlib, _NativeWindow


SOURCE_VERSION = 'research_feature_column_source_v1'
REASONS = (None, 'source_missing', 'not_visible_at_cutoff', 'not_provided',
           'price_missing', 'invalid_price', 'missing_anchor_factor',
           'invalid_anchor_factor', 'missing_factor', 'invalid_factor',
           'invalid_adjusted_value')


def _hash_array(value):
    from hashlib import sha256
    return 'sha256:'+sha256(memoryview(value).cast('B')).hexdigest()


def _source_config(spec):
    from .stock_feature_inputs import _spec
    universe_id=_spec(spec)
    return {'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],
        'symbols':spec['universe'],'universe_id':universe_id,'calendar':spec['calendar'],
        'read_sessions':spec['read_sessions'],'feature_sessions':spec['feature_sessions'],
        'cutoff_by_session':spec['cutoff_by_session'],
        'feature_selection':spec['feature_selection'],'scope_ref':spec['scope']['scope_bundle_ref']}


def feature_column_plan(source, catalog, *, selection=None, normalized=True):
    """Compile the existing catalog DAG once with actual global source bindings."""
    from axiom_engine.core import ABI,SEMANTICS,FeaturePlan
    from .feature_catalog import build_feature_plan
    spec=source.manifest['spec']; a=source.arrays
    selection=spec['feature_selection'] if selection is None else selection
    calendar_ref=digest({'snapshot_id':spec['snapshot'],'sessions':spec['read_sessions']})
    reference_ref=digest({'source_ref':source.identity,'member':source.manifest['buffers']['member'],
        'member_available':source.manifest['buffers']['member_available']})
    header={'abi':ABI,'semantics':SEMANTICS,'recipe_ref':catalog.recipe_ref(selection,normalized=normalized),
        'calendar_ref':calendar_ref,'reference_ref':reference_ref,
        'reference_members':{day:{security:None for j,security in enumerate(spec['universe']) if bool(a['member'][i,j])}
            for i,day in enumerate(spec['read_sessions'])},
        'input_schema':[{'name':f,'dtype':'float64','unit':'CNY' if f=='amount_cny' else 'CNY/share',
            'stage':'fact','missing':'preserve'} for f in FIELDS[:-1]],
        'event_schema':{},'sources':sorted(source.source_bindings(),key=lambda b:b['id']),
        'observation_domain':'sessions','history_policy':'partial','nodes':[],'outputs':[],'obligations':[]}
    return build_feature_plan(FeaturePlan.from_dict(header),selection,catalog=catalog,normalized=normalized)


class FeatureColumnSource:
    """Readonly admitted arrays; daily views share the physical source."""
    def __init__(self, manifest, arrays):
        self.manifest, self.arrays = manifest, arrays
        self.positions = {day:i for i,day in enumerate(manifest['spec']['read_sessions'])}
        for value in arrays.values(): value.flags.writeable=False

    @property
    def identity(self):
        return self.manifest['source_ref']

    def select(self, day, *, history_sessions=252):
        """Apply the original output anchor using the Data-owned array kernel."""
        import numpy as np
        from axiom_data.derived import _adjust_columns
        spec=self.manifest['spec']; end=self.positions[day]+1; begin=end-history_sessions
        require(begin>=0 and day in spec['feature_sessions'],'complete declared output history required')
        a=self.arrays; shape=(history_sessions,len(spec['universe']),4)
        factor=np.broadcast_to(a['values'][begin:end,:,5,None],shape)
        factor_valid=np.broadcast_to(a['validity'][begin:end,:,5,None],shape)
        anchor=np.broadcast_to(a['values'][end-1:end,:,5,None],shape)
        anchor_valid=np.broadcast_to(a['validity'][end-1:end,:,5,None],shape)
        values,validity,reasons=_adjust_columns(a['values'][begin:end,:,:4],
            a['validity'][begin:end,:,:4],a['reasons'][begin:end,:,:4],
            factor,factor_valid,anchor,anchor_valid)
        price_clock=a['available'][begin:end,:,:4]
        price_known=a['available_validity'][begin:end,:,:4]
        factor_clock=np.broadcast_to(a['available'][begin:end,:,5,None],shape)
        factor_known=np.broadcast_to(a['available_validity'][begin:end,:,5,None],shape)
        anchor_clock=np.broadcast_to(a['available'][end-1:end,:,5,None],shape)
        anchor_known=np.broadcast_to(a['available_validity'][end-1:end,:,5,None],shape)
        # Null leaves do not contribute a clock. Absence is established at this
        # output cutoff only when every raw/factor/anchor leaf clock is null.
        sentinel=np.iinfo(np.int64).min
        known=price_known|factor_known|anchor_known
        clock=np.maximum.reduce([np.where(price_known,price_clock,sentinel),
            np.where(factor_known,factor_clock,sentinel),np.where(anchor_known,anchor_clock,sentinel)])
        cutoff=instant_us(spec['cutoff_by_session'][day])//1000000*1000000
        clock=np.where(known,(clock+999999)//1000000*1000000,cutoff)
        amount_known=a['available_validity'][begin:end,:,4]
        amount_clock=np.where(amount_known,(a['available'][begin:end,:,4]+999999)//1000000*1000000,cutoff)
        require(bool((clock<=cutoff).all()) and bool((amount_clock<=cutoff).all()),'future source dependency')
        selection={'source_ref':self.identity,'session':day,'history_sessions':spec['read_sessions'][begin:end],
            'anchor_session':day,'knowledge_cutoff':spec['cutoff_by_session'][day]}
        return {'selection':selection,'selection_ref':digest(selection),
            'sessions':tuple(spec['read_sessions'][begin:end]),'universe':tuple(spec['universe']),
            'values':{name:values[:,:,i] for i,name in enumerate(FIELDS[:4])}|
                     {'amount_cny':a['values'][begin:end,:,4]},
            'validity':{name:validity[:,:,i] for i,name in enumerate(FIELDS[:4])}|
                       {'amount_cny':a['validity'][begin:end,:,4]},
            'availability':{name:clock[:,:,i] for i,name in enumerate(FIELDS[:4])}|
                           {'amount_cny':amount_clock},
            'missing_reasons':{name:reasons[:,:,i] for i,name in enumerate(FIELDS[:4])}|
                              {'amount_cny':a['reasons'][begin:end,:,4]},
            'member':a['member'][begin:end],
            'member_available':np.where(a['member_available_validity'][begin:end],
                (a['member_available'][begin:end]+999999)//1000000*1000000,cutoff)}

    def source_bindings(self):
        """Stable source range IDs; selections bind the exact dependency window."""
        return [dict(id=digest([self.identity,field]),data_ref=digest({'snapshot':self.manifest['spec']['snapshot']}),
            view_ref=self.identity,revision_policy=self.manifest['spec']['pit_policy'],
            qualification='best_effort',availability_basis='declared_vendor_assumption')
            for field in (*FIELDS,'is_member')]

    def core_inputs(self, day, *, calendar_ref, reference_ref, schema, history_sessions=252):
        import numpy as np
        view=self.select(day,history_sessions=history_sessions); shape=view['member'].shape
        ids={field:binding['id'] for field,binding in zip((*FIELDS,'is_member'),self.source_bindings())}
        sets=[[ids[field],ids['factor']] for field in FIELDS[:4]]+[[ids['amount_cny']],[ids['is_member']]]
        sets=[sorted(set(group)) for group in sets]
        inputs=dict(contract_version='feature_column_input_v1',calendar_ref=calendar_ref,
            reference_ref=reference_ref,schema=schema,source_bindings=sorted(self.source_bindings(),key=lambda b:b['id']),
            sessions=list(view['sessions']),security_ids=list(view['universe']),
            cutoffs_utc_us=np.full(len(view['sessions']),instant_us(view['selection']['knowledge_cutoff'])//1000000*1000000,dtype='<i8'),
            values=view['values'],validity=view['validity'],reason_codes={k:v.astype('<i4') for k,v in view['missing_reasons'].items()},
            reason_dictionary=list(REASONS),available_at_utc_us=view['availability'],
            source_codes={field:np.full(shape,i,dtype='<i4') for i,field in enumerate(FIELDS[:-1])},
            source_sets=sets,reference_member=view['member'],reference_industry=np.full(shape,None,dtype=object),
            reference_available_at_utc_us=view['member_available'],reference_source_codes=np.full(shape,5,dtype='<i4'),
            input_refs={'source_ref':self.identity,'selection_ref':view['selection_ref']})
        return inputs,view['selection']


def build_feature_column_source(data, *, spec, destination, block_sessions=20, progress=None):
    """Prepare once with public reads and the unchanged float32 projection."""
    import numpy as np
    import pandas as pd
    from .stock_ml import _implementation, _environment
    require(spec['pit_policy']=='best_effort_vendor_v1','column Feature source needs proved vendor selection invariance')
    require(type(block_sessions) is int and 0<block_sessions<=32,'bounded source calendar block required')
    require(spec['read_sessions']==sorted(set(spec['read_sessions'])) and
        all(instant_us(spec['cutoff_by_session'][b])>=instant_us(spec['cutoff_by_session'][a])
            for a,b in zip(spec['read_sessions'],spec['read_sessions'][1:])),'monotone historical cutoffs required')
    snapshot=data.store.load_snapshot(spec['snapshot'])
    from axiom_data.reader import _release_time
    profiles={}
    for name in ('market_daily','adjustment_factors','universe_membership'):
        domain=snapshot['domains'][name]; profile=domain['source_profile']; profiles[name]={
            'contract':domain['contract'],'source_profile':profile}
        require(all(instant_us(_release_time(profile,day).isoformat())<=instant_us(spec['cutoff_by_session'][day])
            for day in spec['read_sessions']),'hidden later release prevents shared source admission')
    require(profiles['universe_membership']['source_profile']['id']=='tushare.index_weight.dated_membership.v1' and
        profiles['universe_membership']['contract']['contract_id']=='local.tushare_dated_membership.v1',
        'membership source needs stable dated-event identities')
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    config=_source_config(spec); prepared=prepare_matrix_qlib(data,config=config,destination=root)
    sessions=spec['read_sessions']; symbols=spec['universe']; shape=(len(sessions),len(symbols),len(FIELDS))
    native=_NativeWindow(prepared['view'].read(fields=FIELDS,symbols=symbols,
        start=sessions[0],end=sessions[-1]),sessions=sessions,symbols=symbols,
        instrument_map=prepared['view_reference']['instrument_map'])
    arrays={'values':np.zeros(shape,dtype='<f8'),'original_values':np.zeros(shape,dtype='<f8'),
        'validity':np.zeros(shape,dtype=bool),'available':np.zeros(shape,dtype='<i8'),
        'available_validity':np.zeros(shape,dtype=bool),'reasons':np.zeros(shape,dtype='u1'),
        'member':np.zeros(shape[:2],dtype=bool),'member_available':np.zeros(shape[:2],dtype='<i8'),
        'member_available_validity':np.zeros(shape[:2],dtype=bool)}
    chunks=[]; metrics={'data_read_calls':0,'feature_core_calls':0,'source_rows':0,'native_value_reuses':0,
        'reader_projection_fallback_cells':0,'qlib_prepare_seconds':prepared['seconds']}
    clock_cache={}
    def timestamp(value):
        if value not in clock_cache: clock_cache[value]=instant_us(value)
        return clock_cache[value]
    for offset in range(0,len(sessions),block_sessions):
        days=sessions[offset:offset+block_sessions]; keys=[(s,d) for d in days for s in symbols]
        expected=set(keys); chunk_shape=(len(days),len(symbols)); evidence=[]
        for base in (prepared['price_query'],prepared['factor_query'],prepared['member_query']):
            query=replace(base,sessions=tuple(days),cutoff_by_session={d:spec['cutoff_by_session'][d] for d in days})
            batch=data.members(snapshot=spec['snapshot'],query=query) if query.domain=='universe_membership' else data.read(snapshot=spec['snapshot'],query=query)
            metrics['data_read_calls']+=1
            frame=batch.frame.set_index(['security_id','session'])
            require(frame.index.is_unique and set(frame.index)==expected,'complete unique source keys required')
            frame=frame.reindex(pd.MultiIndex.from_tuples(keys,names=['security_id','session']))
            for field in query.fields:
                meta=batch.field_meta[field]; by_key={(r['security_id'],r['session']):r for r in meta['by_key']}
                require(len(by_key)==len(meta['by_key']) and set(by_key)==expected,'complete unique field provenance required')
                ordered=[by_key[key] for key in keys]
                for (security,day),row in zip(keys,ordered):
                    require(row.get('missing_reason')!='not_visible_at_cutoff','later visibility needs a distinct source cohort')
                    require(row.get('availability_basis') in (None,'declared_vendor_assumption'),
                        'unproved source selection invariance')
                    require(row.get('usable_from') is None or timestamp(row['usable_from'])<=timestamp(spec['cutoff_by_session'][day]),
                        'source exceeds historical row cutoff')
                clocks=np.asarray([timestamp(r['usable_from']) if r.get('usable_from') is not None else 0
                    for r in ordered],dtype='<i8').reshape(chunk_shape)
                known=np.asarray([r.get('usable_from') is not None for r in ordered],dtype=bool).reshape(chunk_shape)
                if field=='is_member':
                    require(not frame[field].isna().any(),'unknown membership cannot become false')
                    arrays['member'][offset:offset+len(days)]=frame[field].to_numpy(dtype=bool).reshape(chunk_shape)
                    require(all(r.get('source_snapshot_date') is None or r['source_snapshot_date']<=d
                        for (_,d),r in zip(keys,ordered)),'future membership snapshot')
                    arrays['member_available'][offset:offset+len(days)]=clocks
                    arrays['member_available_validity'][offset:offset+len(days)]=known
                    continue
                require(str(meta['dtype']).lower() in ('float','float32','float64','double','int','int32','int64','integer'),
                    'numeric source field required')
                index=FIELDS.index(field); original=frame[field].to_numpy(dtype=np.float64,na_value=np.nan).reshape(chunk_shape)
                valid=~np.isnan(original)
                with np.errstate(over='ignore',invalid='ignore',under='ignore'):
                    projected=original.astype(np.float32).astype(np.float64)
                require(bool(np.isfinite(projected[valid]).all()),'nonfinite float32 projection')
                native_values=native.frame['$'+field].reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=np.float64).reshape(chunk_shape)
                matches=(~valid&np.isnan(native_values))|(valid&(projected==native_values))
                metrics['native_value_reuses']+=int((matches&valid).sum())
                metrics['reader_projection_fallback_cells']+=int((~matches).sum())
                selected=np.where(matches,native_values,projected)
                reasons=np.asarray([REASONS.index(r.get('missing_reason')) for r in ordered],dtype='u1').reshape(chunk_shape)
                require(bool((reasons[valid]==0).all()),'present source has a missing reason')
                arrays['values'][offset:offset+len(days),:,index]=np.where(valid,selected,0)
                arrays['original_values'][offset:offset+len(days),:,index]=np.where(valid,original,0)
                arrays['validity'][offset:offset+len(days),:,index]=valid
                arrays['available'][offset:offset+len(days),:,index]=clocks
                arrays['available_validity'][offset:offset+len(days),:,index]=known
                arrays['reasons'][offset:offset+len(days),:,index]=reasons
            child=seal({'contract_version':'reader_column_evidence_v1','context':batch.context,
                'field_meta':batch.field_meta,'fields':list(query.fields),'sessions':list(days)},'evidence_ref')
            evidence.append(write_part(root,child,'evidence_ref')); del batch,frame,child,ordered,by_key
        chunks.append({'sessions':list(days),'evidence':evidence}); metrics['source_rows']+=len(keys)
        if progress is not None: progress({'stage':'feature_column_source','completed_dates':offset+len(days),
            'total_dates':len(sessions),**metrics})
    del native
    descriptors={}
    for name,array in arrays.items():
        dtype='bool_u8' if array.dtype.kind=='b' else 'uint8' if array.dtype==np.dtype('u1') else 'int64_le' if array.dtype.kind=='i' else 'float64_le'
        descriptors[name]=write_buffer(root,array.astype('u1',copy=False).reshape(-1) if dtype=='bool_u8' else array.reshape(-1),
            dtype=dtype,shape=list(array.shape))
    implementations=_implementation()
    manifest=seal({'contract_version':SOURCE_VERSION,'spec':spec,'qlib_view':prepared['view_reference'],
        'buffers':descriptors,'chunks':chunks,'reason_codes':list(REASONS),'source_profiles':profiles,
        'selection_invariance':'declared_vendor_release_visible_at_each_historical_row_cutoff',
        'implementation_sources':implementations,'implementation_ref':digest(implementations),
        'environment':_environment()},'source_ref')
    write_json(root/'source.json',manifest)
    return FeatureColumnSource(manifest,arrays),metrics


def load_feature_column_source(path):
    """Hash-verify saved physical inputs without Data reads or computation."""
    import json
    import numpy as np
    path=Path(path); manifest=json.loads(path.read_text()); _verify_ref(manifest,'source_ref')
    require(manifest['contract_version']==SOURCE_VERSION and manifest['reason_codes']==list(REASONS),
        'column source version/reason dictionary mismatch')
    arrays={}; types={'float64_le':'<f8','int64_le':'<i8','bool_u8':'u1','uint8':'u1'}
    for name,desc in manifest['buffers'].items():
        require(file_digest(desc['path'])==desc['file_digest']==desc['buffer_digest'],'column source buffer changed')
        shape=tuple(desc['shape']); require(Path(desc['path']).stat().st_size==int(np.prod(shape))*np.dtype(types[desc['dtype']]).itemsize,
            'column source buffer shape mismatch')
        value=np.fromfile(desc['path'],dtype=types[desc['dtype']]).reshape(shape)
        if desc['dtype']=='bool_u8':
            require(bool(((value==0)|(value==1)).all()),'invalid source bool encoding'); value=value.astype(bool)
        arrays[name]=value
    for chunk in manifest['chunks']:
        for desc in chunk['evidence']:
            require(file_digest(desc['path'])==desc['file_digest'],'column source provenance changed')
    return FeatureColumnSource(manifest,arrays)

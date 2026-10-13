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
        'reference_members':{day:{security:groups[j] for j,security in enumerate(spec['universe']) if bool(a['member'][i,j])}
            for i,day in enumerate(spec['read_sessions']) for groups in source.industries([i])},
        'input_schema':source.input_schema(),
        'event_schema':{},'sources':sorted(source.source_bindings(),key=lambda b:b['id']),
        'observation_domain':'sessions','history_policy':'partial','nodes':[],'outputs':[],'obligations':[]}
    return build_feature_plan(FeaturePlan.from_dict(header),selection,catalog=catalog,normalized=normalized)


class FeatureColumnSource:
    """Readonly admitted arrays; daily views share the physical source."""
    def __init__(self, manifest, arrays):
        self.manifest, self.arrays = manifest, arrays
        self.positions = {day:i for i,day in enumerate(manifest['spec']['read_sessions'])}
        self.security_positions={security:i for i,security in enumerate(manifest['spec']['universe'])}
        for value in arrays.values(): value.flags.writeable=False

    @property
    def identity(self):
        return self.manifest['source_ref']

    def input_schema(self):
        fields=[{'name':f,'dtype':'float64','unit':'CNY' if f=='amount_cny' else 'CNY/share',
                 'stage':'fact','missing':'preserve'} for f in FIELDS[:-1]]
        fields.extend(dict(name=f,dtype='float64',unit='dimensionless' if m['dtype']=='bool' else m['unit'],stage='fact',missing='preserve')
                      for f,m in self.manifest.get('daily_fields',{}).items())
        if 'volume_shares' in self.manifest.get('daily_fields',{}):
            fields.append(dict(name='vwap',dtype='float64',unit='CNY/share',stage='fact',missing='preserve'))
        if 'up_limit' in self.manifest.get('daily_fields',{}):
            fields.extend(dict(name=f,dtype='float64',unit='CNY/share',stage='fact',missing='preserve') for f in ('raw_open','raw_close'))
        fields.append(dict(name='raw_factor',dtype='float64',unit='dimensionless',stage='fact',missing='preserve'))
        fields.append(dict(name='session_ordinal',dtype='float64',unit='days',stage='fact',missing='preserve'))
        return fields

    def physical_fields(self):
        return (*FIELDS, *self.manifest.get('daily_fields',{}), 'is_member')

    def field_array(self, field, kind):
        return (self.arrays[kind][:,:,FIELDS.index(field)] if field in FIELDS else
                self.arrays[field+'::'+kind])

    def leaf_fields(self, field):
        if field=='session_ordinal': return ('close',)
        if field in ('raw_open','raw_close','raw_factor'): return (field[4:],)
        if field == 'vwap': return ('amount_cny','volume_shares','factor')
        return (field,'factor') if field in FIELDS[:4] else (field,)

    def industries(self, rows, securities=None):
        """Decode frozen SW L1 interval inputs; unknown never becomes a group."""
        import numpy as np
        columns=list(range(len(self.manifest['spec']['universe']))) if securities is None else securities
        if 'industry_code' not in self.manifest.get('daily_fields',{}):
            return np.full((len(rows),len(columns)),None,dtype=object)
        values=self.arrays['industry_code::values'][np.ix_(rows,columns)]
        valid=self.arrays['industry_code::validity'][np.ix_(rows,columns)]
        return np.asarray([[f'SW.L1.{int(v)}' if ok else None for v,ok in zip(vs,oks)] for vs,oks in zip(values,valid)],dtype=object)

    def require_coverage(self, fields, securities):
        """Unmaterialized financial securities are outside the source, not null facts."""
        declarations=self.manifest.get('financial_materializations',{})
        for field in fields:
            if field in declarations:
                require(set(securities)<=set(declarations[field]['coverage_security_ids']),
                        'requested securities outside financial field coverage')

    def select(self, day, *, history_sessions=252, securities=None, fields=None):
        """Apply Data's original anchor kernel to just the selected physical block."""
        import numpy as np
        from axiom_data.derived import _adjust_columns
        spec=self.manifest['spec']; end=self.positions[day]+1; begin=end-history_sessions
        require(begin>=0 and day in spec['feature_sessions'],'complete declared output history required')
        securities=list(spec['universe'] if securities is None else securities)
        require(bool(securities) and securities==sorted(set(securities)) and set(securities)<=set(spec['universe']),
                'selected securities outside source')
        fields=list(FIELDS[:-1] if fields is None else fields)
        require(bool(fields) and len(fields)==len(set(fields)) and set(fields)<={c['name'] for c in self.input_schema()},
                'selected fields outside source')
        self.require_coverage(fields,securities)
        positions=[self.security_positions[s] for s in securities]
        # Bound copies by the requested history and securities before adjustment.
        a={name:(value[begin:end] if securities==spec['universe'] else value[begin:end][:,positions])
           for name,value in self.arrays.items() if '::' not in name}
        cutoff=instant_us(spec['cutoff_by_session'][day])//1000000*1000000
        prices=[name for name in fields if name in FIELDS[:4]]; indices=[FIELDS.index(name) for name in prices]
        selected_values={}; selected_validity={}; selected_reasons={}; selected_clocks={}
        if prices:
            shape=(history_sessions,len(securities),len(prices))
            factor=np.broadcast_to(a['values'][:,:,5,None],shape)
            factor_valid=np.broadcast_to(a['validity'][:,:,5,None],shape)
            anchor=np.broadcast_to(a['values'][-1:,:,5,None],shape)
            anchor_valid=np.broadcast_to(a['validity'][-1:,:,5,None],shape)
            values,validity,reasons=_adjust_columns(a['values'][:,:,indices],
                a['validity'][:,:,indices],a['reasons'][:,:,indices],
                factor,factor_valid,anchor,anchor_valid)
            price_clock=a['available'][:,:,indices]; price_known=a['available_validity'][:,:,indices]
            factor_clock=np.broadcast_to(a['available'][:,:,5,None],shape)
            factor_known=np.broadcast_to(a['available_validity'][:,:,5,None],shape)
            anchor_clock=np.broadcast_to(a['available'][-1:,:,5,None],shape)
            anchor_known=np.broadcast_to(a['available_validity'][-1:,:,5,None],shape)
            sentinel=np.iinfo(np.int64).min; known=price_known|factor_known|anchor_known
            clock=np.maximum.reduce([np.where(price_known,price_clock,sentinel),
                np.where(factor_known,factor_clock,sentinel),np.where(anchor_known,anchor_clock,sentinel)])
            clock=np.where(known,(clock+999999)//1000000*1000000,cutoff)
            require(bool((clock<=cutoff).all()),'future source dependency')
            selected_values={name:values[:,:,i] for i,name in enumerate(prices)}
            selected_validity={name:validity[:,:,i] for i,name in enumerate(prices)}
            selected_reasons={name:reasons[:,:,i] for i,name in enumerate(prices)}
            selected_clocks={name:clock[:,:,i] for i,name in enumerate(prices)}
        if 'amount_cny' in fields:
            amount_clock=np.where(a['available_validity'][:,:,4],
                (a['available'][:,:,4]+999999)//1000000*1000000,cutoff)
            require(bool((amount_clock<=cutoff).all()),'future source dependency')
            selected_values['amount_cny']=a['values'][:,:,4]
            selected_validity['amount_cny']=a['validity'][:,:,4]
            selected_reasons['amount_cny']=a['reasons'][:,:,4]
            selected_clocks['amount_cny']=amount_clock
        # Added daily leaves retain their native numeric precision and clocks.
        def leaf(field, kind):
            value=self.field_array(field,kind)[begin:end]
            return value if securities==spec['universe'] else value[:,positions]
        for field in fields:
            if field in FIELDS or field=='vwap': continue
            if field=='session_ordinal':
                from datetime import date
                days=spec['read_sessions'][begin:end]
                selected_values[field]=np.broadcast_to(np.asarray([date.fromisoformat(d).toordinal() for d in days],dtype='float64')[:,None],(len(days),len(securities)))
                selected_validity[field]=np.ones((len(days),len(securities)),dtype=bool)
                selected_reasons[field]=np.zeros((len(days),len(securities)),dtype='u1')
                selected_clocks[field]=np.zeros((len(days),len(securities)),dtype='<i8')
                continue
            if field in ('raw_open','raw_close','raw_factor'):
                native=field[4:]
                clock=np.where(leaf(native,'available_validity'),(leaf(native,'available')+999999)//1000000*1000000,cutoff)
                require(bool((clock<=cutoff).all()),'future native price dependency')
                selected_values[field]=leaf(native,'original_values')
                selected_validity[field]=leaf(native,'validity')
                selected_reasons[field]=leaf(native,'reasons')
                selected_clocks[field]=clock
                continue
            clock=np.where(leaf(field,'available_validity'),
                (leaf(field,'available')+999999)//1000000*1000000,cutoff)
            require(bool((clock<=cutoff).all()),'future source dependency')
            selected_values[field]=leaf(field,'values')
            selected_validity[field]=leaf(field,'validity')
            selected_reasons[field]=leaf(field,'reasons')
            selected_clocks[field]=clock
        if 'vwap' in fields:
            amount=leaf('amount_cny','original_values'); volume=leaf('volume_shares','values')
            valid=leaf('amount_cny','validity') & leaf('volume_shares','validity') & (volume>0) & (amount>0)
            reasons=np.where(~leaf('amount_cny','validity'),leaf('amount_cny','reasons'),
                np.where(~leaf('volume_shares','validity'),leaf('volume_shares','reasons'),
                         np.where(valid,0,REASONS.index('invalid_price')))).astype('u1')
            raw=np.divide(amount,volume,out=np.zeros_like(amount),where=valid)
            factor=leaf('factor','values'); factor_valid=leaf('factor','validity')
            values,validity,reasons=_adjust_columns(raw,valid,reasons,factor,factor_valid,
                np.broadcast_to(factor[-1:],factor.shape),np.broadcast_to(factor_valid[-1:],factor.shape))
            clocks=[]
            for f in ('amount_cny','volume_shares','factor'):
                clocks.append(np.where(leaf(f,'available_validity'),leaf(f,'available'),np.iinfo(np.int64).min))
            clocks.append(np.broadcast_to(clocks[-1][-1:],factor.shape))
            clock=np.maximum.reduce(clocks)
            known=clock!=np.iinfo(np.int64).min
            clock=np.where(known,(np.where(known,clock,0)+999999)//1000000*1000000,cutoff)
            require(bool((clock<=cutoff).all()),'future source dependency')
            selected_values['vwap']=values; selected_validity['vwap']=validity
            selected_reasons['vwap']=reasons; selected_clocks['vwap']=clock
        selection={'source_ref':self.identity,'session':day,'history_sessions':spec['read_sessions'][begin:end],
            'anchor_session':day,'knowledge_cutoff':spec['cutoff_by_session'][day]}
        return {'selection':selection,'selection_ref':digest(selection),
            'sessions':tuple(spec['read_sessions'][begin:end]),'universe':tuple(securities),
            'values':selected_values,'validity':selected_validity,
            'availability':selected_clocks,'missing_reasons':selected_reasons,
            'member':a['member'],
            'industry':self.industries(list(range(begin,end)),positions),
            'member_available':np.where(a['member_available_validity'],
                (a['member_available']+999999)//1000000*1000000,cutoff)}

    def source_bindings(self):
        """Stable source range IDs; selections bind the exact dependency window."""
        return [dict(id=digest([self.identity,field]),data_ref=digest({'snapshot':self.manifest.get('financial_materializations',{}).get(field,{}).get(
                'event_snapshot',self.manifest['spec']['snapshot'])}),
            view_ref=self.identity,revision_policy=self.manifest['spec']['pit_policy'],
            qualification='best_effort',availability_basis=('financial_components_max_declared_vendor_assumption'
                if field in self.manifest.get('financial_materializations',{}) else 'declared_vendor_assumption'))
            for field in self.physical_fields()]

    def core_inputs(self, day, *, calendar_ref, reference_ref, schema, history_sessions=252, securities=None):
        import numpy as np
        view=self.select(day,history_sessions=history_sessions,securities=securities,
                         fields=[c['name'] for c in schema]); shape=view['member'].shape
        ids={field:binding['id'] for field,binding in zip(self.physical_fields(),self.source_bindings())}
        names=[c['name'] for c in self.input_schema()]
        sets=[sorted(ids[f] for f in self.leaf_fields(name)) for name in names]+[[ids['is_member']]]
        inputs=dict(contract_version='feature_column_input_v1',calendar_ref=calendar_ref,
            reference_ref=reference_ref,schema=schema,source_bindings=sorted(self.source_bindings(),key=lambda b:b['id']),
            sessions=list(view['sessions']),security_ids=list(view['universe']),
            cutoffs_utc_us=np.full(len(view['sessions']),instant_us(view['selection']['knowledge_cutoff'])//1000000*1000000,dtype='<i8'),
            values=view['values'],validity=view['validity'],reason_codes={k:v.astype('<i4') for k,v in view['missing_reasons'].items()},
            reason_dictionary=list(REASONS),available_at_utc_us=view['availability'],
            source_codes={field:np.full(shape,i,dtype='<i4') for i,field in enumerate(names) if field in view['values']},
            source_sets=sets,reference_member=view['member'],reference_industry=view['industry'],
            reference_available_at_utc_us=view['member_available'],reference_source_codes=np.full(shape,len(names),dtype='<i4'),
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
    snapshot=data.describe_snapshot(snapshot=spec['snapshot'],
        domains=('market_daily','adjustment_factors','universe_membership'))
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
    from .feature_input_buffers import allocate_feature_buffer, seal_feature_buffer
    arrays={name:allocate_feature_buffer(root,dtype=dtype,shape=shape[:2] if name.startswith('member') else shape)
            for name,dtype in {'values':'<f8','original_values':'<f8','validity':bool,
                'available':'<i8','available_validity':bool,'reasons':'u1',
                'member':bool,'member_available':'<i8','member_available_validity':bool}.items()}
    chunks=[]; metrics={'data_read_calls':0,'feature_core_calls':0,'source_rows':0,'native_value_reuses':0,
        'reader_projection_fallback_cells':0,'qlib_prepare_seconds':prepared['seconds']}
    clock_cache={}
    def timestamp(value):
        if value not in clock_cache: clock_cache[value]=instant_us(value)
        return clock_cache[value]
    for offset in range(0,len(sessions),block_sessions):
        days=sessions[offset:offset+block_sessions]; keys=[(s,d) for d in days for s in symbols]
        native=_NativeWindow(prepared['view'].read(fields=FIELDS,symbols=symbols,
            start=days[0],end=days[-1]),sessions=days,symbols=symbols,
            instrument_map=prepared['view_reference']['instrument_map'])
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
        descriptors[name],arrays[name]=seal_feature_buffer(root,array,dtype=dtype,shape=list(array.shape))
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


def extend_feature_column_source(data, *, source, daily_fields, destination, block_sessions=20,
                                 maximum_added_bytes=512*1024**2, snapshot=None):
    """Append explicitly typed daily facts without reading or rewriting old fields.

    daily_fields maps canonical names to {domain, dtype, unit}. Only daily Data
    queries whose declared releases are visible at every row cutoff are admitted.
    Event histories must use an explicit as-of selector and are not daily fields.
    The returned source retains all original buffer and evidence descriptors.
    """
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    from axiom_data import QuerySpec
    from axiom_data.reader import _release_time
    require(type(block_sessions) is int and 0<block_sessions<=32,'bounded source calendar block required')
    require(bool(daily_fields) and type(daily_fields) is dict,'explicit daily fields required')
    manifest=deepcopy(source.manifest); spec=manifest['spec']
    require(spec['pit_policy']=='best_effort_vendor_v1','daily extension needs proved vendor selection invariance')
    previous=manifest.get('daily_fields',{})
    require(not (set(daily_fields)&set(source.physical_fields())) and 'vwap' not in daily_fields,
            'daily extension cannot replace existing or derived fields')
    for field,meta in daily_fields.items():
        require(type(field) is str and field.isidentifier() and type(meta) is dict and
                set(meta)=={'domain','dtype','unit'},'typed daily field declaration required')
    original_snapshot=spec['snapshot']
    advancing=snapshot is not None and snapshot!=original_snapshot
    requested=sorted({meta['domain'] for meta in daily_fields.values()} |
                     (set(manifest['source_profiles']) if advancing else set()))
    if advancing:
        prior=data.describe_snapshot(snapshot=original_snapshot,
            domains=sorted(manifest['source_profiles']),include_domain_refs=True)
    candidate=data.describe_snapshot(snapshot=snapshot if advancing else original_snapshot,
        domains=requested,include_domain_refs=advancing)
    if advancing:
        reused={domain:metadata['domain_ref'] for domain,metadata in prior['domains'].items()}
        require(all(candidate['domains'][domain]['domain_ref']==ref for domain,ref in reused.items()),
                'candidate snapshot changes reused source domains')
        manifest.setdefault('snapshot_extensions',[]).append({'from_snapshot':original_snapshot,
            'to_snapshot':snapshot,'reused_domains':reused})
        spec['snapshot']=snapshot
    snapshot=candidate; domains={}
    for field,meta in daily_fields.items():
        require(type(field) is str and field.isidentifier() and type(meta) is dict and
                set(meta)=={'domain','dtype','unit'},'typed daily field declaration required')
        require(meta['dtype'] in ('float64','int64','bool'),'numeric daily field dtype required')
        domain=snapshot['domains'][meta['domain']]; contract=domain['contract']; profile=domain['source_profile']
        require(contract['logical_key']==['security_id','session'],'event domains require explicit as-of selection')
        actual=contract['fields'][field]
        require(actual['dtype']==meta['dtype'] and actual.get('unit')==meta['unit'],'daily field dtype/unit mismatch')
        require(all(instant_us(_release_time(profile,d).isoformat())<=instant_us(spec['cutoff_by_session'][d])
                    for d in spec['read_sessions']),'hidden later release prevents daily extension')
        if field=='volume_shares':
            require(meta=={'domain':'market_daily','dtype':'int64','unit':'shares'},'canonical volume shares required')
        domains.setdefault(meta['domain'],[]).append(field)
        incoming={'contract':contract,'source_profile':profile}
        if meta['domain'] in manifest['source_profiles']:
            from .stock_feature_library import SavedColumnFeatureSource
            for existing in source.physical_fields():
                old=manifest['source_profiles'][meta['domain']]
                if existing in old['contract']['fields']:
                    require(SavedColumnFeatureSource._field_profile(old,existing)==
                            SavedColumnFeatureSource._field_profile(incoming,existing),
                            'daily extension changes existing field semantics')
        manifest['source_profiles'][meta['domain']]=incoming
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    days=spec['read_sessions']; symbols=spec['universe']; shape=(len(days),len(symbols))
    require(type(maximum_added_bytes) is int and maximum_added_bytes>0 and
            len(days)*len(symbols)*27*len(daily_fields)<=maximum_added_bytes,
            'daily extension buffers exceed maximum_added_bytes')
    from .feature_input_buffers import allocate_feature_buffer, seal_feature_buffer
    arrays=dict(source.arrays); metrics={'data_read_calls':0,'added_fields':len(daily_fields),'existing_field_reads':0}
    kinds={'values':('<f8','float64_le'),'original_values':('<f8','float64_le'),
        'validity':(bool,'bool_u8'),'available':('<i8','int64_le'),
        'available_validity':(bool,'bool_u8'),'reasons':('u1','uint8')}
    for field in daily_fields:
        for kind,(dtype,_) in kinds.items(): arrays[field+'::'+kind]=allocate_feature_buffer(root,dtype=dtype,shape=shape)
    chunks=[]
    for offset in range(0,len(days),block_sessions):
        selected=days[offset:offset+block_sessions]; keys=[(s,d) for d in selected for s in symbols]
        expected=set(keys); chunk_shape=(len(selected),len(symbols)); evidence=[]
        for domain,fields in domains.items():
            query=QuerySpec(domain,tuple(fields),tuple(symbols),tuple(selected),spec['pit_policy'],
                            {d:spec['cutoff_by_session'][d] for d in selected})
            batch=data.read(snapshot=spec['snapshot'],query=query); metrics['data_read_calls']+=1
            frame=batch.frame.set_index(['security_id','session'])
            require(frame.index.is_unique and set(frame.index)==expected,'complete unique daily keys required')
            frame=frame.reindex(pd.MultiIndex.from_tuples(keys,names=['security_id','session']))
            for field in fields:
                meta=batch.field_meta[field]; declaration=daily_fields[field]
                require(meta['dtype']==declaration['dtype'] and meta['unit']==declaration['unit'],
                        'reader daily field dtype/unit mismatch')
                by_key={(r['security_id'],r['session']):r for r in meta['by_key']}
                require(len(by_key)==len(meta['by_key']) and set(by_key)==expected,'complete unique daily provenance required')
                rows=[by_key[k] for k in keys]
                require(all(r.get('missing_reason')!='not_visible_at_cutoff' and
                    r.get('availability_basis') in (None,'declared_vendor_assumption') and
                    (r.get('usable_from') is None or instant_us(r['usable_from'])<=instant_us(spec['cutoff_by_session'][d]))
                    for (_,d),r in zip(keys,rows)),'unproved daily selection invariance')
                values=frame[field].to_numpy(dtype=np.float64,na_value=np.nan).reshape(chunk_shape)
                valid=~np.isnan(values)
                require(bool(np.isfinite(values[valid]).all()),'nonfinite daily value')
                if declaration['dtype']=='int64':
                    require(bool(((np.abs(values[valid])<=2**53)&(values[valid]==np.floor(values[valid]))).all()),
                            'integer outside exact binary64 range')
                reasons=np.asarray([REASONS.index(r.get('missing_reason')) for r in rows],dtype='u1').reshape(chunk_shape)
                require(bool((reasons[valid]==0).all()) and bool((reasons[~valid]!=0).all()),'daily validity/reason mismatch')
                payload={'values':np.where(valid,values,0),'original_values':np.where(valid,values,0),'validity':valid,
                    'available':np.asarray([instant_us(r['usable_from']) if r.get('usable_from') else 0 for r in rows]).reshape(chunk_shape),
                    'available_validity':np.asarray([r.get('usable_from') is not None for r in rows]).reshape(chunk_shape),
                    'reasons':reasons}
                for kind,value in payload.items(): arrays[field+'::'+kind][offset:offset+len(selected)]=value
            child=seal({'contract_version':'reader_column_evidence_v1','context':batch.context,
                'field_meta':batch.field_meta,'fields':fields,'sessions':selected},'evidence_ref')
            evidence.append(write_part(root,child,'evidence_ref'))
        chunks.append({'sessions':selected,'evidence':evidence})
    for field in daily_fields:
        for kind,(_,dtype) in kinds.items():
            name=field+'::'+kind; value=arrays[name]
            manifest['buffers'][name],arrays[name]=seal_feature_buffer(root,value,dtype=dtype,shape=shape)
    manifest['daily_fields']={**previous,**deepcopy(daily_fields)}
    manifest.setdefault('daily_extensions',[]).append({'fields':list(daily_fields),'chunks':chunks})
    manifest.pop('source_ref'); manifest=seal(manifest,'source_ref')
    write_json(root/'source.json',manifest)
    return FeatureColumnSource(manifest,arrays),metrics

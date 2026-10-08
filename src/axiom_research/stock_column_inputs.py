"""Consume public Data column selections; no price adjustment mathematics.

The public adjusted selection must declare its original query and complete
price/factor/anchor dependency binding. An unadjusted selection is rejected;
there is no fallback to DataBatch/to_json or Data implementation internals.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields as dataclass_fields
import os

from .stock_artifacts import digest
from .stock_fold_inputs import require, seal
from .stock_compact_store import reference, sealed
from .stock_label_contracts import _instant


class ColumnMathReuse:
    """Bounded numerical memo in the current preparation owner, not a store.

    Logical sources and clocks are rebuilt from every actual Data selection.
    Cache entries contain only immutable numerical vectors and Core origin refs.
    """
    def __init__(self,metrics):
        self.metrics=metrics;self.raw={};self.normalized={};self.charge=0

    def _remember(self,cache,key,value):
        from .stock_compact_labels import _working,_sync_feature_charge
        from .stock_compact_store import _size
        old=None
        try:
            old=cache.get(key);amount=_size(value)+sum(a.nbytes for a in value.values() if hasattr(a,'nbytes'))+1024
            old_amount=0 if old is None else old['_charge']
            _working(self.metrics,max(0,amount-old_amount))
            cache[key]={**value,'_charge':amount};self.charge+=amount-old_amount
            state=self.metrics.get('_owner_state')
            if state is not None:
                state._fixed_shared_bytes+=amount-old_amount
                self.metrics['_owner_shared_baseline']+=amount-old_amount
            self.metrics['_column_math_bytes']=self.charge;_sync_feature_charge(self.metrics)
            return cache[key]
        finally:old=value=None

    def close(self):
        self.raw.clear();self.normalized.clear();self.charge=0
        self.metrics['_column_math_bytes']=0

    @staticmethod
    def vector_ref(arrays):
        from hashlib import sha256
        h=sha256()
        for a in arrays:
            h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(memoryview(a))
        return 'sha256:'+h.hexdigest()

    def forward_block(self,keys,opening,closing,valid,*,anchor,dependency_refs,operator):
        """Reuse daily dependencies, execute all changed rows in one Core batch."""
        try:
            import numpy as np
            require(opening.shape==closing.shape==valid.shape and opening.ndim==2 and
                len(keys)==len(dependency_refs)==opening.shape[0], 'Raw block shape mismatch')
            numerics=[self.vector_ref((opening[i],closing[i],valid[i])) for i in range(len(keys))]
            output=[]; changed=[]
            for i,key in enumerate(keys):
                old=self.raw.get(key)
                if old is not None and old['numeric_input_ref']==numerics[i] and old['anchor']==anchor and old['dependency_ref']==dependency_refs[i]:
                    self.metrics['raw_numeric_reuses']=self.metrics.get('raw_numeric_reuses',0)+1
                    output.append(old)
                else:output.append(None);changed.append(i)
            a=b=mask=result=values=flags=vals=flags_day=old=None
            try:
                if changed:
                    from .stock_compact_labels import _working
                    _working(self.metrics,len(changed)*opening.shape[1]*80+4096)
                    # Endpoint gathering is orchestration; divide then subtract remains
                    # exclusively in Core. There is one call for the entire Raw block.
                    a=np.ascontiguousarray(opening[changed].reshape(-1));b=np.ascontiguousarray(closing[changed].reshape(-1))
                    mask=np.ascontiguousarray(valid[changed].reshape(-1))
                    a.flags.writeable=b.flags.writeable=mask.flags.writeable=False
                    result=operator(a,b,endpoint_validity=mask)
                    values=np.asarray(result['values']);flags=np.asarray(result['validity'])
                    require(values.shape==flags.shape==a.shape and values.dtype==np.dtype('<f8') and
                        flags.dtype==np.dtype('?') and not values.flags.writeable and not flags.flags.writeable and
                        np.isfinite(values).all() and not np.any(flags & ~mask) and np.all(values[~flags]==0.0),
                        'Core forward output shape, validity or ownership mismatch')
                    self.metrics['forward_core_calls']=self.metrics.get('forward_core_calls',0)+1
                    self.metrics['core_calls']=self.metrics.get('core_calls',0)+1
                    self.metrics['label_core_calls']=self.metrics.get('label_core_calls',0)+1
                    width=opening.shape[1]
                    for slot,i in enumerate(changed):
                        # Owned bytes per day avoid retaining a large Core block for
                        # one surviving cache row after a moving-window eviction.
                        vals=np.frombuffer(values[slot*width:(slot+1)*width].tobytes(),dtype='<f8')
                        flags_day=np.frombuffer(flags[slot*width:(slot+1)*width].tobytes(),dtype='?')
                        output[i]=self._remember(self.raw,keys[i],{'numeric_input_ref':numerics[i],
                            'anchor':anchor,'dependency_ref':dependency_refs[i],'values':vals,'validity':flags_day})
            except BaseException:
                output.clear();raise
            finally:
                a=b=mask=result=values=flags=vals=flags_day=old=None
                opening=closing=valid=None

            return output

        finally:opening=closing=valid=None

    def normalization(self,day,numeric_ref):
        old=self.normalized.get(day)
        return old if old is not None and old['numeric_input_ref']==numeric_ref else None

    def remember_normalization(self,day,numeric_ref,values,validity,core_ref):
        try:
            return self._remember(self.normalized,day,{'numeric_input_ref':numeric_ref,
                'values':values,'validity':validity,'core_result_ref':core_ref})
        finally:values=validity=None


def _public(value, name):
    return value[name] if isinstance(value,Mapping) else getattr(value,name)


def _plain(value):
    if isinstance(value,Mapping):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def query_binding(query):
    value={f.name:(dict(v) if isinstance(v,Mapping) else list(v) if isinstance(v,tuple) else v)
        for f in dataclass_fields(query) for v in (getattr(query,f.name),)}
    value['cutoff_by_session']={s:_instant(v).isoformat() for s,v in value['cutoff_by_session'].items()}
    return value


def validate_column_raw_binding(definition, common, cutoff):
    """Readonly structural admission of the explicit new source proof."""
    source=definition['price_view'];sealed(source,'price_view_ref')
    from .stock_compact_store import fields
    fields(source,{'contract_version','snapshot_ref','query_binding','selection_ref','source_block_refs',
        'revision_binding','evidence_binding','source_binding','adjustment_binding','input_queries',
        'column_metadata','price_basis','anchor_session','price_view_ref'},'exact column price source proof required')
    require(source['contract_version']=='stock_label_column_price_view_v1' and
        source['snapshot_ref']==common['snapshot'] and reference(source['selection_ref']) and
        type(source['source_block_refs']) is list and bool(source['source_block_refs']) and
        all(reference(r) for r in source['source_block_refs']) and
        source['price_basis']=='common_anchor_adjusted_v1', 'column price source proof mismatch')
    metadata=source['column_metadata']
    require(type(metadata) is dict and set(metadata)=={'open','close'} and
        all(type(v) is dict and v.get('dtype')=='float64' and
            v.get('basis')==v.get('price_basis')=='common_anchor_adjusted_v1' and
            v.get('recipe_version')=='common_anchor_price_v1' and
            type(v.get('unit')) is str and bool(v['unit']) for v in metadata.values()) and
        metadata['open']['unit']==metadata['close']['unit'],'column price dtype/unit/basis mismatch')
    q=source['query_binding'];h=common['target_spec']['horizon_sessions']
    require(definition['calendar']==common['calendar'] and definition['universe']==common['universe'] and
        definition['snapshot']==common['snapshot'] and _instant(definition['cutoff'])==_instant(cutoff) and
        definition['horizon_sessions']==h and type(h) is int and h>0 and
        definition['start_session_offset']==1 and definition['end_session_offset']==h and
        definition['formula']==f'close(f+{h}) / open(f+1) - 1' and
        definition['label_definition_ref']==common['target_spec']['label_definition_ref'] and
        definition['price_basis']=='common_anchor_adjusted_v1' and
        definition['missing_policy']=='invalid_null_preserve_grid', 'column Raw target/calendar mismatch')
    require(q['domain']=='market_daily' and q['fields']==['open','close'] and
        q['symbols']==common['universe'] and q['pit_policy']==common['pit_policy'] and
        q['purpose']=='label_outcomes' and q['price_basis']=='common_anchor_adjusted_v1' and
        q['adjustment_anchor']==source['anchor_session'] and
        set(q['cutoff_by_session'])==set(q['sessions']) and
        all(_instant(v)==_instant(cutoff) for v in q['cutoff_by_session'].values()),
        'column source original QuerySpec mismatch')
    allowed=[d for d in common['calendar'] if d<=_instant(cutoff).date().isoformat()]
    require(bool(allowed) and source['anchor_session']==allowed[-1], 'column adjustment anchor mismatch')
    positions={day:i for i,day in enumerate(common['calendar'])}
    expected={allowed[-1]}
    for day in definition['sessions']:
        for off in (1,h):
            n=positions[day]+off
            if n<len(common['calendar']) and common['calendar'][n]<=allowed[-1]:expected.add(common['calendar'][n])
    require(expected<=set(q['sessions']) and set(q['sessions'])<=set(common['calendar']),
            'column source endpoint coverage mismatch')
    require(type(source['adjustment_binding']) is dict and bool(source['adjustment_binding']),
            'Data adjusted price/factor/anchor binding required')
    derivation=source['adjustment_binding']
    require(derivation['recipe_version']=='common_anchor_price_v1' and
        derivation['formula']=='price_t * factor_t / factor_anchor' and derivation['factor_field']=='factor' and
        derivation['anchor_session']==derivation['decision_session']==source['anchor_session'] and
        _instant(derivation['decision_cutoff'])==_instant(cutoff) and
        all(reference(derivation[k]) for k in ('price_selection_ref','factor_selection_ref')),
        'column adjustment recipe/source/clock mismatch')
    expected={**q,'price_basis':'unadjusted','adjustment_anchor':None}
    require(set(source['input_queries'])=={'price','factor'} and source['input_queries']['price']==expected and
        source['input_queries']['factor']=={**expected,'domain':'adjustment_factors','fields':['factor']},
        'column adjustment original input queries mismatch')
    require(source['source_binding']['snapshot_ref']==common['snapshot'] and
        source['source_binding']['domain']=='market_daily' and reference(source['source_binding']['contract_ref']) and
        source['source_binding']['source_block_refs']==source['source_block_refs'] and
        source['revision_binding']['source_block_refs']==source['source_block_refs'] and
        set(source['revision_binding']['indices'])=={'index:price','index:factor','index:anchor_factor'} and
        all(reference(r) for r in source['revision_binding']['indices'].values()) and
        source['evidence_binding']['selected_versions']==source['revision_binding'] and
        all(reference(r) for r in source['evidence_binding']['source_block_refs']),
        'column source/revision/evidence closure mismatch')


class ColumnPriceDomain:
    """One public adjusted panel and small immutable source controls."""
    def __init__(self, selection, *, spec, query, anchor, input_queries,metrics=None):
        self.panels={};self.lineage={};self.positions={};self.metrics=None;self.source=None
        self.selection=self.spec=None;self.closed=False;self.pid=os.getpid()
        try:self._initialize(selection,spec=spec,query=query,anchor=anchor,input_queries=input_queries,metrics=metrics)
        except BaseException:
            self.close();raise

    def _initialize(self, selection, *, spec, query, anchor, input_queries,metrics=None):
        column=panels=a=borrow=None
        try:
            import numpy as np
            self.pid=os.getpid();self.closed=False;self.selection=selection;self.metrics=metrics
            self.spec=spec;axes=_public(selection,'axes')
            self.positions={day:i for i,day in enumerate(axes['sessions'])}
            require(_public(selection,'contract_version')=='data_column_selection_v1' and
                _public(selection,'snapshot_ref')==spec['snapshot'] and
                list(axes['sessions'])==list(query.sessions) and
                list(axes['security'])==spec['universe'] and
                list(axes['fields'])==['open','close'], 'public column selection axes/source mismatch')
            actual=_plain(_public(selection,'query_binding'))
            require(actual==query_binding(query),'public adjusted selection must bind the exact requested QuerySpec')
            self.source=seal({'contract_version':'stock_label_column_price_view_v1',
                'snapshot_ref':_public(selection,'snapshot_ref'),'query_binding':deepcopy(actual),
                'selection_ref':_public(selection,'selection_ref'),
                'source_block_refs':list(_public(selection,'source_block_refs')),
                'revision_binding':_plain(_public(selection,'revision_binding')),
                'evidence_binding':_plain(_public(selection,'evidence_binding')),
                'source_binding':_plain(_public(selection,'source_binding')),
                'adjustment_binding':_plain(_public(selection,'derivation')),
                'input_queries':_plain(input_queries),
                'column_metadata':{name:_plain(_public(_public(selection,'columns')[name],'metadata'))
                    for name in ('open','close')},
                'price_basis':'common_anchor_adjusted_v1','anchor_session':anchor},'price_view_ref')
            require(reference(self.source['selection_ref']) and bool(self.source['adjustment_binding']),
                    'public adjusted selection lacks actual derivation/source refs')
            metadata=self.source['column_metadata']
            require(all(v.get('dtype')=='float64' and v.get('basis')==v.get('price_basis')=='common_anchor_adjusted_v1' and
                v.get('recipe_version')=='common_anchor_price_v1' and type(v.get('unit')) is str and bool(v['unit'])
                for v in metadata.values()) and metadata['open']['unit']==metadata['close']['unit'],
                'public adjusted price dtype/unit/basis mismatch')
            self.panels={}; shape=(len(query.sessions),len(query.symbols))
            self.lineage={}
            for name,borrow in _public(selection,'selected_version_indices').items():
                a=borrow.to_numpy(dtype='<i8')
                require(a.shape==(*shape,2) and not a.flags.writeable,'column lineage shape/ownership mismatch')
                self.lineage[name]=a
            require(set(self.lineage)=={'price','factor','anchor_factor'},'adjusted endpoint lineage required')
            for name in ('open','close'):
                column=_public(selection,'columns')[name]
                require(_public(column,'metadata')['price_basis']=='common_anchor_adjusted_v1',
                        'Research cannot derive or relabel unadjusted Data prices')
                # Public to_numpy transfers owned, readonly primitive copies. The
                # Data owner reserves copy workspace; our producer charges copies
                # for their subsequent lifetime. No private array is accessed.
                panels={key:_public(column,key).to_numpy(dtype='datetime64[us]' if key=='available_at' else None)
                        for key in ('values','validity','available_at')}
                panels['available_at']=panels['available_at'].view('<i8')
                for key,a in panels.items():
                    require(a.size==shape[0]*shape[1] and not a.flags.writeable,
                            'public selection buffers must be readonly and complete')
                    panels[key]=a.reshape(shape)
                require(panels['values'].dtype==np.dtype('float64') and
                        panels['validity'].dtype==np.dtype('?') and
                        panels['available_at'].dtype==np.dtype('int64'),
                        'column prices require native float64/bool/UTC-microsecond clocks')
                panels['missing_reason']=_public(column,'missing_reason')
                self.panels[name]=panels
        finally:column=panels=a=borrow=selection=None

    @property
    def source_ref(self):return self.source['price_view_ref']

    def _endpoint_dependency(self,indices,clocks):
        """Canonicalize source slots without a per-cell text provenance graph."""
        import numpy as np
        refs=self.source['source_block_refs'];used=sorted({refs[int(n)] for a in indices
            for n in np.unique(a[...,0]) if n>=0})
        slots={ref:i for i,ref in enumerate(used)};arrays=[]
        for a in indices:
            b=a.copy()
            for n in np.unique(a[...,0]):
                if n>=0:b[...,0][a[...,0]==n]=slots[refs[int(n)]]
            arrays.append(b)
        return digest({'blocks':used,'indices_and_clocks':ColumnMathReuse.vector_ref((*arrays,*clocks))})

    def rows(self, days, *, horizon, forward_operator, reuse=None,role='training'):
        """Gather a native endpoint block, qualify it, and call Core once."""
        import numpy as np
        from .stock_matrix_storage import instant_us
        from .stock_compact_batch import _clock
        from .stock_compact_labels import _working
        require(not self.closed and self.pid==os.getpid(),'column price domain is closed or foreign')
        opening=closing=valid=reasons=clocks=values=flags=panel=take=present=safe=code=mask=a=None
        gathered=endpoint_indices=endpoint_clocks=lineage=output=result=None
        try:
            calendar=self.spec['calendar'];positions={d:i for i,d in enumerate(calendar)}
            width=len(self.spec['universe']);shape=(len(days),width)
            cutoff=instant_us(next(iter(self.source['query_binding']['cutoff_by_session'].values())))
            if self.metrics is not None:_working(self.metrics,len(days)*width*240+65536)
            starts=[calendar[positions[d]+1] if positions[d]+1<len(calendar) else None for d in days]
            ends=[calendar[positions[d]+horizon] if positions[d]+horizon<len(calendar) else None for d in days]
            # Codes preserve the original start-before-end reason precedence. An
            # actual endpoint outside the as-of selection is missing, whereas an
            # endpoint beyond the frozen calendar is calendar_endpoint_uncovered.
            dictionary=[None,'calendar_endpoint_uncovered']
            for label in ('start_open','end_close'):
                dictionary.extend(('missing_'+label,'invalid_'+label,'nonpositive_'+label,
                    'unknown_'+label+'_availability','unavailable_'+label+':provenance_exceeds_query_cutoff'))
            reasons=np.zeros(shape,dtype='u1');gathered=[];endpoint_indices=[];endpoint_clocks=[]
            for field,endpoints,base in (('open',starts,2),('close',ends,7)):
                take=np.asarray([self.positions.get(d,-1) for d in endpoints],dtype='<i8')
                present=take>=0;safe=np.maximum(take,0);panel=self.panels[field]
                values=panel['values'][safe].copy();flags=panel['validity'][safe].copy()
                clocks=panel['available_at'][safe].copy()
                flags &= present[:,None]
                code=np.zeros(shape,dtype='u1')
                tests=((~flags,base),(~np.isfinite(values),base+1),(values<=0,base+2),
                    (clocks==np.iinfo(np.int64).min,base+3),(clocks>cutoff,base+4))
                for mask,n in tests:code[(code==0)&mask]=n
                code[np.asarray([d is None for d in endpoints])]=1
                reasons[(reasons==0)&(code!=0)]=code[(reasons==0)&(code!=0)]
                # Source dependency identities include selected physical versions,
                # endpoint lineage clocks and source block refs, not whole query
                # cutoffs or unrelated selected records.
                lineage=[]
                for name in ('price','factor','anchor_factor'):
                    a=self.lineage[name][safe].copy();a[~present]=-1;lineage.append(a)
                endpoint_indices.append(lineage);endpoint_clocks.append(clocks)
                gathered.append(values)
            valid=reasons==0;opening,closing=gathered
            opening[~valid]=closing[~valid]=0.0
            for a in (opening,closing,valid):a.flags.writeable=False
            dependencies=[self._endpoint_dependency(
                [a[i] for endpoint in endpoint_indices for a in endpoint],
                [a[i] for a in endpoint_clocks]) for i in range(len(days))]
            if reuse is None:
                result=forward_operator(opening.reshape(-1),closing.reshape(-1),endpoint_validity=valid.reshape(-1))
                values=np.asarray(result['values']).reshape(shape);flags=np.asarray(result['validity']).reshape(shape)
                output=[{'values':values[i],'validity':flags[i]} for i in range(len(days))]
            else:output=reuse.forward_block([(role,d,horizon) for d in days],opening,closing,valid,
                anchor=self.source['anchor_session'],dependency_refs=dependencies,operator=forward_operator)
            clocks=np.maximum(*endpoint_clocks)
            for i,day in enumerate(days):
                values=output[i]['values'];flags=output[i]['validity']
                require(values.shape==flags.shape==(width,) and not values.flags.writeable and not flags.flags.writeable,
                        'Core forward output shape/ownership mismatch')
                for j,security in enumerate(self.spec['universe']):
                    ok=bool(flags[j]);reason=dictionary[int(reasons[i,j])] if not valid[i,j] else None if ok else 'nonfinite_return'
                    yield {'security_id':security,'feature_session':day,'start_session':starts[i],'end_session':ends[i],
                        'return':float(values[j]) if ok else None,'label_available_at':_clock(clocks[i,j]) if ok else None,
                        'valid':ok,'invalid_reason':reason,'source_refs':[self.source_ref]}
        finally:
            opening=closing=valid=reasons=clocks=values=flags=panel=take=present=safe=code=mask=a=None
            gathered=endpoint_indices=endpoint_clocks=lineage=output=result=None

    def close(self):
        self.panels.clear();self.lineage.clear();self.positions.clear();self.source=None
        self.selection=self.spec=None;self.closed=True
        if self.metrics is not None:self.metrics['_price_view_bytes']=0
        self.metrics=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

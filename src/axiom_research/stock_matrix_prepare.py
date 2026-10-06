"""Explicit bounded preparation; numerical targets remain owned by Data/Core.

Saved readers never call this module. Queries retain each fit's native cutoff,
and the scheduler visits adjacent date blocks before advancing across history.
"""
from array import array
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import errno
import struct
import sys
import tempfile
import time

from .stock_artifacts import digest, file_digest, write_json, _read
from .stock_fold_inputs import require, seal, validate_spec, file_fingerprint, read_parent
from .stock_label_contracts import _eligible_reason, _instant, NORMALIZATION_SPEC
from .stock_matrix_storage import (write_buffer, write_part, write_partition,
                                   instant_us)

RAW_SCHEMA = [{'name':'raw_return','dtype':'float64','unit':'dimensionless',
               'stage':'fact','missing':'preserve'}]
NORMALIZED_SCHEMA = [{'name':'normalized_target','dtype':'float64','unit':'dimensionless',
                      'stage':'cross_sectional','missing':'preserve'}]
CORE_TYPES = {'values':('float64','float64_le','d'),
    'value_validity':('bool','bool_u8','?'), 'value_reason_codes':('int32','int32_le','i'),
    'fact_available_at_utc_us':('int64','int64_le','q'),
    'reference_member':('bool','bool_u8','?'),
    'reference_available_at_utc_us':('int64','int64_le','q'),
    'selection_cutoff_utc_us':('int64','int64_le','q'),
    'fact_source_codes':('int32','int32_le','i'),
    'reference_source_codes':('int32','int32_le','i'),
    'available_at_utc_us':('int64','int64_le','q'), 'source_codes':('int32','int32_le','i')}


def _options(value):
    require(type(value) is dict and set(value)=={'row_block_sessions','column_block',
        'maximum_resident_bytes','normalization_backend'}, 'exact preparation options required')
    require(value['normalization_backend']=='core_cs_batch_v1' and all(
        type(value[k]) is int and value[k]>0 for k in value if k!='normalization_backend'),
        'positive preparation budgets and fixed Core backend required')
    return deepcopy(value)


class _Publisher:
    """Write staging bytes with immutable final paths before one directory rename."""
    def __init__(self, stage, target): self.stage,self.target=Path(stage),Path(target)
    def _desc(self, desc):
        desc=deepcopy(desc)
        desc['path']=str((self.target/Path(desc['path']).relative_to(self.stage.resolve())).resolve())
        return desc
    def part(self, value, key): return self._desc(write_part(self.stage,value,key))
    def buffer(self, values, *, dtype, shape):
        return self._desc(write_buffer(self.stage,values,dtype=dtype,shape=shape))
    def actual(self, desc): return self.stage/Path(desc['path']).relative_to(self.target.resolve())


class _V1FeatureAccess:
    """Compatibility bridge: at most one original Feature shard is borrowed."""
    def __init__(self, saved):
        from .stock_matrix_storage import row_index
        self.path=saved.path; self._index=saved.to_dict(); self._spec=self._index['definition']['spec']
        self._row_index=row_index(self._spec['feature_sessions'],self._spec['universe'])
        self._parents={d:p for p in self._index['feature_parents'] for d in p['sessions']}
        self._current=None; self._rows=None; self._marks={}
        self._store=self
        for p in self._index['feature_parents']:
            for k in ('features','input_evidence'):
                self._marks[p[k]['path']]=file_fingerprint(p[k]['path'])
        self._marks[str((self.path/'index.json').resolve())]=file_fingerprint(self.path/'index.json')
    def to_dict(self): self.check(); return deepcopy(self._index)
    @property
    def identity(self): return self._index['feature_inputs_ref']
    @property
    def descriptor(self):
        path=self.path/'index.json'
        return {'path':str(path.resolve()),'file_digest':file_digest(path),'feature_inputs_ref':self.identity}
    def check(self):
        require(all(file_fingerprint(p)==mark for p,mark in self._marks.items()),'original Feature source changed')
    def close(self): self._rows=None; self._current=None
    def row_metadata(self, offsets):
        self.check(); out=[]; ids=self._row_index['security_ids']; days=self._row_index['sessions']; width=len(ids)
        for offset in offsets:
            day=days[int(offset)//width]; security=ids[int(offset)%width]; parent=self._parents[day]
            if parent['features']!=self._current:
                value=read_parent(parent['features'],'feature_ref')
                self._rows={(r['security_id'],r['session']):r for r in value['rows']}
                self._current=parent['features']
            out.append(deepcopy(self._rows[security,day]))
        self.check(); return out


def _v1_partitions(feature, publisher, options):
    """Pack original shards without creating a fictitious new Feature index."""
    from .stock_feature_inputs import _write_feature_matrix_block
    spec=feature._spec; index=feature.to_dict(); partitions=[]; selections=[]
    schema=[{'name':name,'dtype':'float64','unit':'dimensionless','stage':'cross_sectional',
             'missing':'preserve'} for name in spec['ordered_features']]
    rid=publisher.part(feature._row_index,'row_index_ref'); width=len(spec['universe'])
    for parent in index['feature_parents']:
        original=read_parent(parent['features'],'feature_ref'); proof=_read(parent['input_evidence']['path'])
        offset=spec['feature_sessions'].index(parent['sessions'][0])*width
        chunks,selection=_write_feature_matrix_block(publisher.stage,rows=original['rows'],proof=proof,
            spec=spec,view=index['qlib_view'],schema=schema,index_ref=feature._row_index['row_index_ref'],
            row_offset=offset,options={'column_block':options['column_block']})
        for chunk in chunks:
            chunk={k:v for k,v in chunk.items() if k!='partition_ref'}
            chunk['buffers']={k:publisher._desc(v) for k,v in chunk['buffers'].items()}
            chunk['metadata']=publisher._desc(chunk['metadata'])
            partitions.append(seal(chunk,'partition_ref'))
        selections.extend(selection)
    return partitions,selections,schema,rid


def _canonical_buffers(value, publisher, names):
    canonical={k:deepcopy(v) for k,v in value.items() if k not in names}; physical={}
    for name in names:
        dtype,physical_dtype,code=CORE_TYPES[name]; view=memoryview(value[name])
        physical[name]=publisher.buffer((bool(v) if dtype=='bool' else v for v in view),
                                        dtype=physical_dtype,shape=[len(view)])
        canonical[name]={'dtype':dtype,'shape':[len(view)],'bytes_digest':physical[name]['buffer_digest']}
    return canonical,physical


def _readonly(values, name):
    """Only encode values; no arithmetic or eligibility is hidden in this codec."""
    code=CORE_TYPES[name][2]
    return memoryview(struct.pack('<'+str(len(values))+code,*values)).cast(code)


def _clock_text(us):
    from datetime import timedelta
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=int(us))).isoformat().replace('+00:00','Z')


def _graph_bytes(value, seen=None):
    """Conservative owned Python graph accounting; process-tree RSS is separate."""
    seen=set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value)); size=sys.getsizeof(value)
    if isinstance(value,dict): size+=sum(_graph_bytes(k,seen)+_graph_bytes(v,seen) for k,v in value.items())
    elif isinstance(value,(list,tuple,set)): size+=sum(_graph_bytes(v,seen) for v in value)
    return size


def _price_workgroups(data, requests, *, snapshot,budget,stats):
    """Consecutive related cutoffs with a bounded number of retained batches.

    The limit reserves two thirds for the factor/adjusted/label working set.
    It changes grouping only, never query semantics or the numerical cohort.
    """
    group=[]; retained=0; largest_per_row=0.0; limit=max(1,budget//3)
    for request in requests:
        query=request[4]; rows=len(query.sessions)*len(query.symbols)
        if group and retained+largest_per_row*rows>limit:
            yield group,retained; group=[]; retained=0
        price=data.read(snapshot=snapshot,query=query); stats['data_read_calls']+=1
        size=_graph_bytes(price.field_meta)+_graph_bytes(price.context)+int(price.frame.memory_usage(deep=True).sum())
        require(size<=limit,'one Data query working set exceeds resident budget; reduce row_block_sessions')
        if group and retained+size>limit:
            yield group,retained; group=[]; retained=0
        group.append((request,price)); retained+=size
        largest_per_row=max(largest_per_row,size/max(1,rows))
        stats['maximum_retained_price_bytes']=max(stats.get('maximum_retained_price_bytes',0),retained)
    if group: yield group,retained


def _label_partition(publisher, *, table, spec_ref, index_ref, offset, rows,
                     raw_desc, cutoff, contents, core_refs=()):
    normalized=table=='training_normalized_labels'
    value_key='normalized_return' if normalized else 'return'
    valid_key='normalized_valid' if normalized else 'valid'
    clock_key='normalized_available_at' if normalized else 'label_available_at'
    metadata=seal({'contract_version':'stock_matrix_label_metadata_v1',
        'role':'evaluation' if table=='evaluation_raw_labels' else 'training',
        'fold_spec_ref':spec_ref,'cutoff':cutoff,'rows':rows,'raw_build':raw_desc,
        'core_result_refs':list(core_refs),'contents':contents},'metadata_ref')
    buffers={
        'values':publisher.buffer((r[value_key] if r[valid_key] else 0.0 for r in rows),
                                  dtype='float64_le',shape=[len(rows),1]),
        'value_validity':publisher.buffer((r[valid_key] for r in rows),dtype='bool_u8',shape=[len(rows),1]),
        'available_at_utc_us':publisher.buffer((instant_us(r[clock_key]) if r[clock_key] is not None else 0
                        for r in rows),dtype='int64_le',shape=[len(rows),1]),
        'available_at_validity':publisher.buffer((r[clock_key] is not None for r in rows),
                                                 dtype='bool_u8',shape=[len(rows),1])}
    schema=NORMALIZED_SCHEMA if normalized else RAW_SCHEMA
    return write_partition(publisher.stage,table=table,fold_spec_ref=spec_ref,
        row_index_ref=index_ref,schema_digest=digest(schema),row_offset=offset,row_count=len(rows),
        columns=[schema[0]['name']],buffers=buffers,metadata=publisher.part(metadata,'metadata_ref'))


def _query_for(spec, cutoff, days):
    from axiom_data import QuerySpec
    calendar=spec['calendar']; last=_instant(cutoff).date().isoformat()
    allowed=[d for d in calendar if spec['read_sessions'][0]<=d<=last]
    require(bool(allowed),'no actual label calendar at cutoff')
    anchor=allowed[-1]; positions={d:i for i,d in enumerate(calendar)}; wanted={anchor}
    for day in days:
        i=positions[day]
        wanted.update(calendar[i+n] for n in (1,5) if i+n<len(calendar) and calendar[i+n]<=last)
    sessions=tuple(sorted(wanted))
    return QuerySpec('market_daily',('open','close'),tuple(spec['universe']),sessions,
        spec['pit_policy'],{d:cutoff for d in sessions},purpose='label_outcomes'),anchor


def _source_contents(wire, raw, cohort=None):
    # Actual adjusted Query/selected endpoint provenance is retained, rather
    # than claiming a logical whole-window DataBatch that was never queried.
    query=wire['context']['query']; selected={'context':wire['context'],
        'records':wire['records'],'field_meta':wire['field_meta']}
    contents={digest(query):query,digest(selected):selected}
    if cohort is not None: contents[digest(cohort)]=cohort
    return contents,digest(query),digest(selected),digest(cohort) if cohort is not None else None


def _selector(publisher, *, view_ref, row_index, schema_ref, role, fold_ref, offsets):
    ids=row_index['security_ids']; days=row_index['sessions']; width=len(ids)
    offsets=list(offsets)
    require(offsets==sorted(set(offsets)) and all(0<=i<row_index['row_count'] for i in offsets),
            'ordered unique selector offsets required')
    value=seal({'contract_version':'stock_matrix_selector_v1','prepared_view_ref':view_ref,
        'row_index_ref':row_index['row_index_ref'],'schema_digest':schema_ref,'role':role,
        'fold_spec_ref':fold_ref,'encoding':'offsets_u64_le','row_count':len(offsets),
        'payload':publisher.buffer(offsets,dtype='uint64_le',shape=[len(offsets)]),
        'keys_digest':digest([[ids[i%width],days[i//width]] for i in offsets])},'selector_ref')
    return publisher.part(value,'selector_ref')


def prepare_stock_ml_batch_inputs(data, *, feature_inputs, fold_specs, destination,
                                 preparation_options, metrics=None, progress=None):
    """Prepare saved targets through existing public Data and Core exactly once.

    Date blocks are outermost; related fit cutoffs are consecutive within each
    physical working range. cache0 stays the caller's profile. No supplier,
    Feature execution, model training, predictions or account is performed.
    """
    from .stock_feature_inputs import load_stock_feature_inputs
    from .stock_matrix_reader import load_feature_matrix_index
    from .stock_ml import _implementation, _environment
    from .labels import build_forward_labels
    from axiom_data import adjust_prices
    from axiom_engine.core import execute_cs_zscore_batch
    begin=time.perf_counter(); options=_options(preparation_options); read_work=False
    path=feature_inputs.path if hasattr(feature_inputs,'path') else Path(feature_inputs)
    # A public object is a locator, never a trusted skip-validation marker.
    locator=_read(Path(path)/'index.json')
    feature=(load_feature_matrix_index(path) if locator.get('contract_version')=='stock_feature_inputs_v2'
             else _V1FeatureAccess(load_stock_feature_inputs(path)))
    try:
        index=feature.to_dict(); spec=index['definition']['spec']; row_index=feature._row_index
        require(type(fold_specs) is list and bool(fold_specs),'nonempty ordered fold specs required')
        folds=deepcopy(fold_specs); requests=[]; states={}; previous=None
        for fold in folds:
            training,inference=validate_spec(fold,spec['calendar']); ref=digest(fold)
            require(ref not in states and (previous is None or fold['oos_trade_sessions'][0]>previous),
                    'ordered disjoint OOS folds required')
            previous=fold['oos_trade_sessions'][-1]
            require(set(training+inference)<=set(row_index['sessions']),'Feature input does not cover fold')
            states[ref]={'spec':fold,'training':training,'inference':inference,'chunks':[],
                'arrays':{name:array('d' if name=='values' else 'B' if name in
                    ('value_validity','reference_member') else 'i' if name.endswith('codes') else 'q')
                    for name in list(CORE_TYPES)[:9]},'sources':{},'reasons':[],
                'training_offsets':[],'core_refs':[]}
            requests.extend([(ref,'training',training,fold['fit_cutoff']),
                             (ref,'evaluation',inference,fold['evaluation_cutoff'])])
        implementations=_implementation(); definition={'version':'axiom.stock_ml_batch_inputs/2',
            'feature_inputs':feature.descriptor,'fold_specs':folds,'preparation_options':options,
            'implementation_sources':implementations,'implementation_ref':digest(implementations),
            'environment':_environment()}
        definition_ref=digest(definition); target=(Path(destination)/definition_ref[7:]).resolve()
        if target.exists():
            saved=_read(target/'batch.json')
            require(saved['definition']==definition,'cached matrix batch definition mismatch')
            from .stock_batch import load_stock_ml_batch_inputs
            with load_stock_ml_batch_inputs(saved): pass
            if metrics is not None: metrics.update(cache_hit=True,data_read_calls=0,core_calls=0,
                feature_core_calls=0,train_calls=0,predict_calls=0,account_calls=0,total_seconds=time.perf_counter()-begin)
            return saved
        target.parent.mkdir(parents=True,exist_ok=True)
        stats={'cache_hit':False,'data_read_calls':0,'supplier_calls':0,'feature_core_calls':0,
            'label_core_calls':0,'core_calls':0,'train_calls':0,'predict_calls':0,'account_calls':0,
            'date_working_groups':0,'normalization_seconds':0.0,'data_cache_profile':'caller_unchanged',
            'physical_partition_reads':None,'physical_index_hits':None,
            'physical_metrics_reason':'public Data API does not expose physical Reader counters'}
        with tempfile.TemporaryDirectory(prefix='.stock-matrix-',dir=target.parent) as temporary:
            stage=Path(temporary)/'complete'; stage.mkdir(); publisher=_Publisher(stage,target)
            if index['contract_version']=='stock_feature_inputs_v2':
                partitions=list(index['partitions']); feature_selection=_read(index['source_selection']['path'])['feature_rows']
                feature_schema=index['schema']; row_desc=index['row_index']
            else:
                partitions,feature_selection,feature_schema,row_desc=_v1_partitions(feature,publisher,options)
            label_selection=[]; core_wrappers=[]
            needed=sorted({d for _,_,days,_ in requests for d in days}); width=len(spec['universe'])
            positions={d:i for i,d in enumerate(row_index['sessions'])}
            for start in range(0,len(needed),options['row_block_sessions']):
                working=needed[start:start+options['row_block_sessions']]; stats['date_working_groups']+=1
                active=[]
                for ref,role,days,cutoff in requests:
                    wanted=[d for d in working if d in days]
                    # Split at gaps in the fixed row index, preserving exact
                    # partition ranges even if fit windows do not all overlap.
                    groups=[]
                    for day in wanted:
                        if not groups or positions[day]!=positions[groups[-1][-1]]+1: groups.append([])
                        groups[-1].append(day)
                    for group in groups:
                        query,anchor=_query_for(spec,cutoff,group)
                        active.append((ref,role,cutoff,group,query,anchor))
                # Fixed symbols/columns across fits. Market is read for all
                # related cutoffs before factor reads advance this working set.
                read_work=True
                for group,retained in _price_workgroups(data,active,snapshot=spec["snapshot"],
                        budget=options["maximum_resident_bytes"],stats=stats):
                    for request,price in group:
                        ref,role,cutoff,days,query,anchor=request
                        factors=data.read(snapshot=spec['snapshot'],query=replace(query,domain='adjustment_factors',fields=('factor',)))
                        stats['data_read_calls']+=1
                        adjusted=adjust_prices(price,factors,fields=('open','close'),anchor_session=anchor,
                                                decision_session=anchor,factor_field='factor')
                        wire=adjusted.to_json(); raw=build_forward_labels(adjusted,calendar=spec['calendar'],feature_sessions=days)
                        require(retained+_graph_bytes(wire)+_graph_bytes(raw)<=options['maximum_resident_bytes'],
                                'adjusted Label working set resident byte budget exceeded')
                        raw_desc=publisher.part(raw,'label_ref'); rows=raw['rows']; offset=positions[days[0]]*width
                        cohort=None
                        if role=='training':
                            offsets=range(offset,offset+len(rows)); features=feature.row_metadata(offsets)
                            reasons=[_eligible_reason(r,f,len(spec['ordered_features']),_instant(cutoff)) for r,f in zip(rows,features)]
                            for reason,f in zip(reasons,features):
                                if reason is None:
                                    require(all(a is None or _instant(a)<=_instant(cutoff) for a in f['availability']) and
                                        any(a is not None for a in f['availability']),'training Feature clock exceeds fit')
                            cohort={'contract_version':'stock_matrix_label_cohort_v1','feature_inputs_ref':feature.identity,
                                'raw_label_ref':raw['label_ref'],'fold_spec_ref':ref,'cutoff':cutoff,
                                'sessions':days,'keys':[[r['security_id'],r['feature_session']] for r in rows],
                                'eligibility_reasons':reasons,'eligible_keys':[[r['security_id'],r['feature_session']]
                                    for r,reason in zip(rows,reasons) if reason is None]}
                        contents,qref,vref,cref=_source_contents(wire,raw,cohort)
                        raw_partition=_label_partition(publisher,table='training_raw_labels' if role=='training'
                            else 'evaluation_raw_labels',spec_ref=ref,index_ref=row_index['row_index_ref'],offset=offset,
                            rows=rows,raw_desc=raw_desc,cutoff=cutoff,contents=contents)
                        partitions.append(raw_partition)
                        label_selection.append({'role':role,'fold_spec_ref':ref,'cutoff':cutoff,'sessions':days,
                            'adjustment_anchor':anchor,'query_refs':[qref],'selected_versions_ref':vref,'cohort_ref':cref})
                        if role=='training':
                            state=states[ref]
                            for day in days:
                                # Sources are the original saved Raw build and a
                                # pre-Core Feature/cohort ref, never the final view.
                                state['sources'][day]={'bindings':[
                                    {'id':'offline_eligibility','data_ref':feature.identity,'view_ref':cref,
                                     'revision_policy':'frozen_feature_membership_and_explicit_outcome_cutoff',
                                     'qualification':'observed','availability_basis':'derived_offline_cutoff_selection'},
                                    {'id':'raw_labels','data_ref':raw['label_ref'],'view_ref':raw['label_ref'],
                                     'revision_policy':'frozen_saved_label_build','qualification':'observed',
                                     'availability_basis':'exact_raw_label_available_at'}],
                                    'source_sets':[['offline_eligibility'],['raw_labels']]}
                            state['chunks'].append((offset,len(rows),raw_desc,cutoff,raw_partition['metadata']))
                        del wire,raw,rows,contents,adjusted,factors
                    group.clear()
                if progress is not None: progress({'stage':'raw_labels','completed':min(start+len(working),len(needed)),
                    'total':len(needed),'logical_query_count':stats['data_read_calls'],'seconds':time.perf_counter()-begin})
            for ref,state in states.items():
                arrays=state['arrays']; normalized_chunks=[]
                # Rehydrate only this fit's packed input. All other fit source
                # graphs remain on disk; no fold-count multiplier of raw panels.
                for offset,count,raw_desc,cutoff,metadata_desc in state['chunks']:
                    raw=_read(publisher.actual(raw_desc)); metadata=_read(publisher.actual(metadata_desc))
                    cohort=next(v for v in metadata['contents'].values() if type(v) is dict and
                                v.get('contract_version')=='stock_matrix_label_cohort_v1')
                    core_start=len(arrays['values'])
                    arrays['selection_cutoff_utc_us'].extend(instant_us(cutoff) for d in cohort['sessions'])
                    for row,reason in zip(raw['rows'],cohort['eligibility_reasons']):
                        valid=reason is None; arrays['values'].append(float(row['return']) if valid else 0.0)
                        arrays['value_validity'].append(valid); arrays['value_reason_codes'].append(0)
                        arrays['fact_available_at_utc_us'].append(instant_us(row['label_available_at'] if valid else cutoff))
                        arrays['reference_member'].append(valid); arrays['reference_available_at_utc_us'].append(instant_us(cutoff))
                        arrays['fact_source_codes'].append(1 if valid else 0); arrays['reference_source_codes'].append(0)
                    state['reasons'].extend(cohort['eligibility_reasons'])
                    normalized_chunks.append((offset,count,raw_desc,cutoff,metadata_desc,core_start))
                    require(_graph_bytes(state)<=options['maximum_resident_bytes'],'Core input resident byte budget exceeded')
                    del raw,metadata,cohort
                reasons=[None,*sorted({r for r in state['reasons'] if r is not None})]
                codes={r:i for i,r in enumerate(reasons)}
                arrays['value_reason_codes']=array('i',(codes[r] for r in state['reasons']))
                carrier={'contract_version':'core_cs_zscore_batch_input_v1',
                    'calendar_ref':digest({'contract_version':'stock_label_calendar_v1','sessions':spec['calendar']}),
                    'schema':RAW_SCHEMA,'output_schema':NORMALIZED_SCHEMA,'sessions':state['training'],
                    'security_ids':spec['universe'],'reason_dictionary':reasons,
                    'source_bindings_by_session':state['sources'],
                    **{name:_readonly(values,name) for name,values in arrays.items()}}
                native=sum(v.nbytes for k,v in carrier.items() if k in arrays)
                # Core captures nine buffers, then allocates output lists and
                # five owned outputs. Bound these simultaneous copies before
                # entering its public mathematical executor.
                core_budget=_graph_bytes(state)+3*native+len(arrays['values'])*256
                require(core_budget<=options['maximum_resident_bytes'],'Core working set resident byte budget exceeded')
                input_doc,input_buffers=_canonical_buffers(carrier,publisher,arrays)
                core_input={'input':input_doc,'buffers':input_buffers}
                clock=time.perf_counter(); result=execute_cs_zscore_batch(carrier,params=NORMALIZATION_SPEC['params'])
                stats['normalization_seconds']+=time.perf_counter()-clock; stats['core_calls']+=1; stats['label_core_calls']+=1
                output_names=['values','value_validity','value_reason_codes','available_at_utc_us','source_codes']
                canonical,physical=_canonical_buffers(result,publisher,output_names)
                wrapper=seal({'contract_version':'stock_matrix_core_result_v1','result':canonical,
                    'buffers':physical},'core_result_artifact_ref')
                core_wrappers.append(publisher.part(wrapper,'core_result_artifact_ref'))
                core_ref=result['metadata']['result_ref']; state['core_refs']=[core_ref]
                for offset,count,raw_desc,cutoff,metadata_desc,core_start in normalized_chunks:
                    raw=_read(publisher.actual(raw_desc)); metadata=_read(publisher.actual(metadata_desc))
                    contents=metadata['contents']; normalized=[]
                    for local,row in enumerate(raw['rows']):
                        i=core_start+local; valid=bool(result['value_validity'][i]); reason=state['reasons'][i]
                        normalized.append({**row,'normalized_return':float(result['values'][i]) if valid else None,
                            'normalized_valid':valid,'normalized_available_at':_clock_text(result['available_at_utc_us'][i]) if valid else None,
                            'normalization_reason':None if valid else reason or 'NORMALIZATION_UNDEFINED',
                            'normalization_source_refs':sorted(set(row['source_refs']+[raw['label_ref'],feature.identity,
                                state['sources'][row['feature_session']]['bindings'][0]['view_ref'],core_ref]))})
                        if valid: state['training_offsets'].append(offset+local)
                    contents={**contents,digest(core_input):core_input}
                    partitions.append(_label_partition(publisher,table='training_normalized_labels',spec_ref=ref,
                        index_ref=row_index['row_index_ref'],offset=offset,rows=normalized,raw_desc=raw_desc,
                        cutoff=cutoff,contents=contents,core_refs=[core_ref]))
                    del raw,metadata,contents,normalized
                # Release native input copies before the next Core execution.
                state['arrays'].clear(); state['reasons'].clear(); del carrier,result
            selection=seal({'contract_version':'stock_matrix_source_selection_v1','feature_inputs_ref':feature.identity,
                'feature_rows':feature_selection,'label_rows':label_selection},'source_selection_ref')
            schema={'features':feature_schema,'training_raw_labels':RAW_SCHEMA,
                'training_normalized_labels':NORMALIZED_SCHEMA,'evaluation_raw_labels':RAW_SCHEMA}
            view_definition={k:deepcopy(spec[k]) for k in ('scope','snapshot','pit_policy','calendar','universe',
                'catalog_ref','feature_selection','ordered_features')}
            view_definition.update({k:deepcopy(definition[k]) for k in ('feature_inputs','fold_specs',
                'preparation_options','implementation_sources','implementation_ref','environment')})
            view=seal({'contract_version':'stock_ml_prepared_view_v1','definition':view_definition,
                'definition_ref':digest(view_definition),'schema':schema,'schema_digest':digest(schema),
                'row_index':row_desc,'source_selection':publisher.part(selection,'source_selection_ref'),
                'partitions':partitions,'core_results':core_wrappers},'prepared_view_ref')
            view_desc=publisher.part(view,'prepared_view_ref'); outputs=[]
            for ref,state in states.items():
                infer=[positions[d]*width+i for d in state['inference'] for i in range(width)]
                selectors={role:None if role=='validation' else _selector(publisher,view_ref=view['prepared_view_ref'],
                    row_index=row_index,schema_ref=view['schema_digest'],role=role,fold_ref=ref,
                    offsets=state['training_offsets'] if role in ('training','training_labels') else infer)
                    for role in ('training','validation','inference','training_labels','evaluation_labels')}
                inputs=seal({'contract_version':'stock_ml_saved_inputs_v2','prepared_view':view_desc,
                    'fold_spec_ref':ref,'selectors':selectors,'core_result_refs':state['core_refs']},'input_ref')
                outputs.append({'input_manifest':inputs,'fold_spec':state['spec']})
            batch={'contract_version':'stock_ml_batch_inputs_v2','definition':definition,'definition_ref':definition_ref,
                'prepared_view':view_desc,'folds':outputs,'status':'COMPLETE'}
            batch['batch_ref']=digest(batch); batch=seal(batch,'content_digest'); write_json(stage/'batch.json',batch)
            from .stock_matrix_reader import _validate_staged_matrix_batch
            stats['staged_validation']=_validate_staged_matrix_batch(batch,stage=stage,target=target)
            feature._store.check()
            try: stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
                require(_read(target/'batch.json')==batch,'concurrent prepared batch conflict')
        # The complete closure was admitted before rename. Only exact staged
        # bytes are published; repeat full disk admission belongs to the next
        # explicit public batch load or to a future HIT.
        require(_read(target/'batch.json')==batch,'published prepared batch changed')
        stats.update(total_seconds=time.perf_counter()-begin,prepared_view_ref=view['prepared_view_ref'],
                     batch_ref=batch['batch_ref'],logical_query_count=stats['data_read_calls'])
        if metrics is not None: metrics.update(stats)
        return batch
    finally:
        feature.close()
        if read_work and callable(getattr(data,'clear_cache',None)): data.clear_cache()

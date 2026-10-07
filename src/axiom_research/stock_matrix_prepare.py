"""One compact v3 producer; Raw belongs to Research, prices to Data, CS to Core.

The private historical codec helpers serve frozen compatibility fixtures and
old reader options. New production preparation invokes only the v3 entry below.
"""
from array import array
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import errno
import json
import struct
import sys
import tempfile
import time

from .stock_artifacts import digest, file_digest, write_json, _read
from .stock_fold_inputs import require, seal, validate_spec, file_fingerprint, read_parent
from .stock_label_contracts import (_eligible_reason, _instant, NORMALIZATION_SPEC,
    RAW_TARGET_SCHEMA as RAW_SCHEMA, NORMALIZED_TARGET_SCHEMA as NORMALIZED_SCHEMA)
from .stock_matrix_storage import (write_buffer, write_part, write_partition,
                                   instant_us)

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
    required={'row_block_sessions','column_block','maximum_resident_bytes','normalization_backend'}
    require(type(value) is dict and required <= set(value) <= required|{'maximum_source_bytes','maximum_parent_bytes'},
        'exact preparation options required')
    require(value['normalization_backend']=='core_cs_batch_v1' and all(
        type(value[k]) is int and value[k]>0 for k in value if k!='normalization_backend'),
        'positive preparation budgets and fixed Core backend required')
    return deepcopy(value)


class _Publisher:
    """Write staging bytes with immutable final paths before one directory rename."""
    def __init__(self, stage, target, *, maximum_resident_bytes=512*1024**2,
                 maximum_source_bytes=8*1024**3,maximum_parent_bytes=64*1024**2,
                 known_source_bytes=0,metrics=None):
        from .stock_matrix_reader import VerifiedMatrixStore
        self.stage,self.target=Path(stage),Path(target)
        self.maximum=maximum_resident_bytes; self.metrics={} if metrics is None else metrics
        self.live_bytes=0
        require(type(known_source_bytes) is int and 0<=known_source_bytes<=maximum_source_bytes,
                'prepared known source byte budget exceeded')
        self.known_source_bytes=known_source_bytes; self.written={}; self.written_bytes=0
        self.store=VerifiedMatrixStore(maximum_matrix_bytes=self.maximum,maximum_source_bytes=maximum_source_bytes,
                                      maximum_parent_bytes=maximum_parent_bytes,_path_resolver=self.resolve,
                                      _caller_retained_bytes=lambda:self.live_bytes+_graph_bytes(self.written))
    def _desc(self, desc):
        desc=deepcopy(desc)
        physical=Path(desc['path']); path=str(physical.resolve()); size=physical.stat().st_size
        if path not in self.written:
            require(self.known_source_bytes+self.written_bytes+size<=self.store.maximum_source_bytes,
                    'prepared cumulative source byte budget exceeded')
            self.written[path]=size; self.written_bytes+=size
            self.metrics['publisher_unique_written_bytes']=self.written_bytes
            self.metrics['publisher_known_source_bytes']=self.known_source_bytes
        desc['path']=str((self.target/Path(desc['path']).relative_to(self.stage.resolve())).resolve())
        return desc
    def resolve(self,path):
        path=Path(path)
        try: return self.stage/path.relative_to(self.target.resolve())
        except ValueError: return path
    def _bindings(self):
        return tuple(pair for pairs in self.store._native_bindings.values() for pair in pairs)
    def _wire_bindings(self):
        return tuple(pair for pairs in self.store._wire_bindings.values() for pair in pairs)
    def _retained(self,value):
        return self.store._native_caller_bytes(value)
    def native_digest(self,value,*,exclude_ref_key=None):
        from .stock_native_json import native_digest
        bindings=self._bindings(); wires=self._wire_bindings()
        retained=self._retained(value)+sys.getsizeof(bindings)+sys.getsizeof(wires)
        self.store.check()
        result=native_digest(value,coverage_bindings=bindings,wire_bindings=wires,exclude_ref_key=exclude_ref_key,
            maximum_workspace_bytes=self.maximum,caller_retained_bytes=lambda:retained,
            metrics=self.metrics,path_resolver=self.resolve)
        self.store.check(); return result
    def seal(self,value,key):
        value=dict(value); value[key]=self.native_digest(value,exclude_ref_key=key)
        return value
    def part(self, value, key):
        from .stock_native_json import REFS,make_native_carrier
        descriptor=None
        if REFS.get(value.get('contract_version'))==key:
            bindings=self._bindings(); wires=self._wire_bindings()
            retained=self._retained(value)+sys.getsizeof(bindings)+sys.getsizeof(wires)
            descriptor=make_native_carrier(self.stage,value,key,maximum_source_bytes=self.store.maximum_source_bytes,
                maximum_parent_bytes=self.store.maximum_parent_bytes,maximum_workspace_bytes=self.maximum,
                caller_retained_bytes=lambda:retained,metrics=self.metrics,
                descriptor_mapper=self._desc,path_resolver=self.resolve,coverage_bindings=bindings,
                wire_bindings=wires,externalize_label_wire=True)
        return self._desc(write_part(self.stage,value,key) if descriptor is None else descriptor)
    def read_part(self,descriptor,key):
        require(self.store._native_caller_bytes()<=self.maximum,
                'prepared source admission resident budget exceeded')
        value=self.store.read_json(descriptor,key,_revisit=True)
        require(self.store._native_caller_bytes()<=self.maximum,
                'prepared source admission resident budget exceeded')
        return value
    def release_part(self,descriptor,key): self.store.drop_json(descriptor,key)
    def finish(self):
        self.store.check()
        for key,value in self.store.metrics.items():
            if key.startswith(('native_','carrier_','coverage_','label_wire_','peak_workspace','peak_combined_workspace')):
                if isinstance(value,dict):
                    counts=self.metrics.setdefault(key,{})
                    for ref,count in value.items(): counts[ref]=counts.get(ref,0)+count
                elif 'peak' in key: self.metrics[key]=max(self.metrics.get(key,0),value)
                else: self.metrics[key]=self.metrics.get(key,0)+value
        self.metrics['carrier_builder_source_bytes']=self.store.metrics['source_bytes']
        self.store.close()
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
            row_offset=offset,options=options,caller_retained_bytes=lambda:
                _graph_bytes([original,proof,index,spec,schema,partitions,selections]),
            metrics=publisher.metrics,descriptor_mapper=publisher._desc,path_resolver=publisher.resolve)
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
    metadata=publisher.seal({'contract_version':'stock_matrix_label_metadata_v1',
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


def _request_ranges(requests, working, positions):
    """The one physical split rule used by preparation and its observer plan."""
    for ref,role,days,cutoff in requests:
        wanted=[d for d in working if d in days]; groups=[]
        for day in wanted:
            if not groups or positions[day]!=positions[groups[-1][-1]]+1: groups.append([])
            groups[-1].append(day)
        for group in groups:
            yield ref,role,cutoff,group


def _label_work_plan(spec, fold_specs, row_block_sessions):
    """Pure counts for the exact physical plan; no Data/Feature/Label execution."""
    require(type(row_block_sessions) is int and row_block_sessions>0,'positive row block sessions required')
    requests=[]
    for fold in fold_specs:
        training,inference=validate_spec(fold,spec['calendar']); ref=digest(fold)
        requests.extend([(ref,'training',training,fold['fit_cutoff']),
                         (ref,'evaluation',inference,fold['evaluation_cutoff'])])
    needed=sorted({d for _,_,days,_ in requests for d in days})
    positions={d:i for i,d in enumerate(spec['feature_sessions'])}; width=len(spec['universe'])
    raw_calls=raw_rows=query_rows=maximum_query_rows=0
    for start in range(0,len(needed),row_block_sessions):
        for ref,role,cutoff,days in _request_ranges(requests,needed[start:start+row_block_sessions],positions):
            query,_=_query_for(spec,cutoff,days)
            raw_calls+=1; raw_rows+=len(days)*width
            count=len(query.sessions)*width; query_rows+=count; maximum_query_rows=max(maximum_query_rows,count)
    return {'row_block_sessions':row_block_sessions,'raw_calls':raw_calls,'data_calls':2*raw_calls,
        'core_calls':len(fold_specs),'raw_rows':raw_rows,'selected_query_rows':query_rows,
        'maximum_query_rows':maximum_query_rows,'complete_business_grid':True}


def _source_contents(wire, raw, cohort=None, *, digest_fn=digest):
    # Actual adjusted Query/selected endpoint provenance is retained, rather
    # than claiming a logical whole-window DataBatch that was never queried.
    query=wire['context']['query']; selected={'context':wire['context'],
        'records':wire['records'],'field_meta':wire['field_meta']}
    query_ref=digest(query); selected_ref=digest_fn(selected)
    contents={query_ref:query,selected_ref:selected}
    if cohort is not None: contents[digest(cohort)]=cohort
    return contents,query_ref,selected_ref,digest(cohort) if cohort is not None else None


def _selector_payload(publisher, *, row_index, offsets):
    """Save an active fit's offsets before releasing its working graph."""
    ids=row_index['security_ids']; days=row_index['sessions']; width=len(ids)
    h=sha256(); h.update(b'['); previous=-1; count=0
    def checked():
        nonlocal previous,count
        for offset in offsets:
            require(type(offset) is int and previous<offset<row_index['row_count'],
                    'ordered unique selector offsets required')
            if count: h.update(b',')
            h.update(json.dumps([ids[offset%width],days[offset//width]],
                separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
            previous=offset; count+=1
            yield offset
    payload=publisher.buffer(checked(),dtype='uint64_le',shape=[len(offsets)])
    h.update(b']')
    return {'row_count':count,'payload':payload,'keys_digest':'sha256:'+h.hexdigest()}


def _selector(publisher, *, view_ref, row_index, schema_ref, role, fold_ref, payload):
    value=seal({'contract_version':'stock_matrix_selector_v1','prepared_view_ref':view_ref,
        'row_index_ref':row_index['row_index_ref'],'schema_digest':schema_ref,'role':role,
        'fold_spec_ref':fold_ref,'encoding':'offsets_u64_le',**payload},'selector_ref')
    return publisher.part(value,'selector_ref')


def prepare_stock_ml_batch_inputs(data, *, feature_inputs, fold_specs, destination,
                                 preparation_options, metrics=None, progress=None):
    """The only fresh label producer is compact v3; old files load readonly."""
    from .stock_compact_labels import prepare_compact_batch
    return prepare_compact_batch(data,feature_inputs=feature_inputs,fold_specs=fold_specs,
        destination=destination,preparation_options=preparation_options,metrics=metrics,progress=progress)

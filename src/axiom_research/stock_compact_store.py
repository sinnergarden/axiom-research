"""Owned bytes and sealed Feature table views; no numerical/source replay.

Legacy native identities are historical references. Only the actual table
index, typed buffers and row metadata are ingress dependencies here.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo
import json
import os
import sys
import time

from .stock_artifacts import digest, digest_array_rows
from .stock_fold_inputs import require, ordered, file_fingerprint, seal
from .stock_label_contracts import _instant, _session
from .stock_matrix_storage import DTYPES, BUFFER_FIELDS, instant_us

DEFAULT_LIMITS = {'maximum_source_bytes':8*1024**3,
                  'maximum_matrix_bytes':512*1024**2,
                  'maximum_parent_bytes':64*1024**2}
_TOKEN = object()
_VIEWS = WeakKeyDictionary()
_ADMISSION_CLOCK_CACHE_ENTRIES = 1024


def limits(value=None):
    require(type(value) is dict or value is None,'byte limits must be a mapping')
    out={**DEFAULT_LIMITS,**(value or {})}
    require(set(out)==set(DEFAULT_LIMITS) and all(type(v) is int and v>0 for v in out.values()),
            'positive compact byte limits required')
    return out


def fields(value, expected, message):
    require(type(value) is dict and set(value)==set(expected),message)


def reference(value):
    return type(value) is str and len(value)==71 and value.startswith('sha256:') and all(c in '0123456789abcdef' for c in value[7:])


def sealed(value, key):
    try:
        valid=reference(value.get(key)) and value[key]==digest({k:v for k,v in value.items() if k!=key})
    finally:
        value=None
    require(valid,'compact '+key+' identity mismatch')


def _json_pairs(items):
    out={}
    for key,value in items:
        require(key not in out,'duplicate compact JSON key'); out[key]=value
    return out


def _json_constant(value):
    raise ValueError('nonfinite compact JSON constant: '+value)


def owned_path(descriptor,root):
    require(type(descriptor) is dict and reference(descriptor.get('file_digest')) and
            type(descriptor.get('path')) is str and Path(descriptor['path']).is_absolute() and
            Path(descriptor['path']).resolve().is_relative_to(Path(root).resolve()),
            'compact file escaped its owner directory or lacks byte digest')


class OwnedStore:
    """Each admission hashes the immutable bytes it actually decodes/uses.

    No readonly mmap is used: Python bytes backs every NumPy buffer. Stat
    identities detect lifecycle changes; they never replace the initial hash.
    """
    def __init__(self, budgets=None, resolver=None, shared_bytes=0, shared_source_bytes=0):
        self.limits=limits(budgets); self.resolve=resolver or (lambda p:Path(p))
        self.owner_pid=os.getpid()
        self.shared_bytes=shared_bytes; self.shared_source_bytes=shared_source_bytes
        self.resident_bytes=0; self.lease_bytes=0
        self.borrowers=0; self.closed=False; self.marks={}; self.hashes={}; self.arrays={}; self.json={}
        self.charges={}
        self.check_hook=None
        self.metrics={'file_hash_calls':0,'hash_bytes':0,'source_bytes':0,'json_decode_calls':0,
                      'owned_buffer_bytes':0,'mmap_opens':0,'fold_projection_calls':0,'largest_parent_bytes':0,
                      'source_stat_calls':0,'lifecycle_check_calls':0,
                      'released_buffer_bytes':0,'released_json_bytes':0,'peak_resident_bytes':0}

    def _check_owner(self):
        require(self.owner_pid==os.getpid(),'compact store belongs to another process')

    @property
    def maximum_matrix_bytes(self): return self.limits['maximum_matrix_bytes']

    def reserve(self, amount):
        self._check_owner()
        require(type(amount) is int and amount>=0 and
                self.shared_bytes+self.resident_bytes+self.lease_bytes+amount<=self.maximum_matrix_bytes,
                'compact resident byte budget exceeded')
        self.metrics['peak_resident_bytes']=max(self.metrics['peak_resident_bytes'],
            self.shared_bytes+self.resident_bytes+self.lease_bytes+amount)

    def read(self, descriptor, *, retain=False, parent=False):
        self._check_owner()
        require(not self.closed,'compact store is closed')
        path=descriptor['path']; expected=descriptor.get('file_digest')
        require(type(path) is str and Path(path).is_absolute() and (expected is None or reference(expected)),
                'fixed compact file descriptor required')
        physical=self.resolve(path)
        require(not physical.is_symlink() and physical.is_file(),'compact source must be a regular file')
        if path in self.hashes:
            self.check_path(path)
            require(expected is None or expected==self.hashes[path],'conflicting compact file descriptor')
            if path in self.arrays or path in self.json: return None
        # A released leaf must be hashed again, while its original lifecycle
        # fingerprint continues to guard this owner's immutable source.
        if path in self.marks: self.check_path(path)
        mark=file_fingerprint(physical)
        self.metrics['source_stat_calls']+=1
        require(self.shared_source_bytes+self.metrics['source_bytes']+mark[2]<=self.limits['maximum_source_bytes'],
                'compact source byte budget exceeded')
        if parent:
            require(mark[2]<=self.limits['maximum_parent_bytes'],'compact parent byte budget exceeded')
            self.metrics['largest_parent_bytes']=max(self.metrics['largest_parent_bytes'],mark[2])
        self.reserve(mark[2]*(12 if parent else 1)+4096)
        payload=None
        try:
            with physical.open('rb') as stream:
                require(_stat(os.fstat(stream.fileno()))==mark,'compact source changed before read')
                payload=stream.read(mark[2]+1)
                require(len(payload)==mark[2] and _stat(os.fstat(stream.fileno()))==mark and
                        file_fingerprint(physical)==mark,'compact source changed during read')
                self.metrics['source_stat_calls']+=1
            actual='sha256:'+sha256(payload).hexdigest()
            require(expected is None or actual==expected,'compact file digest mismatch')
        except BaseException:
            payload=None
            raise
        self.marks[path]=mark; self.hashes[path]=actual
        self.metrics['file_hash_calls']+=1; self.metrics['hash_bytes']+=len(payload)
        self.metrics['source_bytes']+=len(payload)
        return payload

    def read_json(self, descriptor, *, key=None, legacy=False, keep=True):
        value=payload=None
        try:
            self._check_owner()
            path=descriptor['path']
            if path in self.json:
                self.check_path(path); value=self.json[path]
                require(descriptor.get('file_digest',self.hashes[path])==self.hashes[path],
                        'conflicting compact JSON descriptor')
            else:
                payload=self.read(descriptor,parent=True)
                require(payload is not None,'released compact JSON cannot be revisited')
                value=json.loads(payload,object_pairs_hook=_json_pairs,parse_constant=_json_constant)
                self.metrics['json_decode_calls']+=1
                if legacy and value.get('contract_version') in ('stock_native_json_carrier_v1','stock_native_json_carrier_v2'):
                    # Runtime row facts live in the verified skeleton. Coverage/
                    # wire slots and historical canonical hashes are audit-only.
                    value=value['skeleton']
                if keep:
                    charge=_size(value,maximum=self.maximum_matrix_bytes,
                        retained=self.shared_bytes+self.resident_bytes+self.lease_bytes+len(payload))
                    self.reserve(charge); self.resident_bytes+=charge
                    self.json[path]=value; self.charges[path]=charge
            if key: sealed(value,key)
            return value
        except BaseException:
            value=payload=None
            raise

    def buffer(self, descriptor):
        array=payload=None
        try:
            self._check_owner()
            fields(descriptor,BUFFER_FIELDS,'exact compact buffer descriptor required')
            require(descriptor['dtype'] in DTYPES and descriptor['file_digest']==descriptor['buffer_digest'] and
                    type(descriptor['shape']) is list and bool(descriptor['shape']) and
                    all(type(v) is int and v>=0 for v in descriptor['shape']), 'compact buffer dtype/shape required')
            path=descriptor['path']; import numpy as np
            shape=tuple(descriptor['shape']); count=1
            for size in shape: count*=size
            dtype={'float64_le':'<f8','bool_u8':'u1','int64_le':'<i8','uint64_le':'<u8',
                   'int32_le':'<i4','uint8':'u1'}[descriptor['dtype']]
            if path not in self.arrays:
                payload=self.read(descriptor,retain=True)
                require(len(payload)==count*np.dtype(dtype).itemsize,'compact buffer byte length mismatch')
                self.reserve(len(payload)); self.resident_bytes+=len(payload)
                self.metrics['owned_buffer_bytes']+=len(payload)
                self.arrays[path]=payload; self.charges[path]=len(payload)
            else: self.check_path(path)
            require(self.hashes[path]==descriptor['file_digest'],'conflicting compact buffer digest')
            payload=self.arrays[path]
            require(len(payload)==count*np.dtype(dtype).itemsize,'conflicting compact buffer shape')
            array=np.frombuffer(payload,dtype=dtype).reshape(shape)
            require(not array.flags.writeable,'owned bytes must remain readonly')
            if descriptor['dtype']=='bool_u8': require(bool(((array==0)|(array==1)).all()),'compact bool bytes must be 0/1')
            return array
        except BaseException:
            array=payload=None
            raise

    def release(self, paths):
        """Drop owned cache references; callers first drop their array views."""
        self._check_owner(); require(not self.closed and self.borrowers==0,
                                    'compact backing still borrowed')
        released={'buffer_bytes':0,'json_bytes':0}
        for path in set(paths):
            kind='buffer_bytes' if path in self.arrays else 'json_bytes'
            charge=self.charges.pop(path,0)
            self.arrays.pop(path,None); self.json.pop(path,None); self.hashes.pop(path,None)
            self.resident_bytes-=charge; released[kind]+=charge
        require(self.resident_bytes>=0,'compact release accounting mismatch')
        self.metrics['released_buffer_bytes']+=released['buffer_bytes']
        self.metrics['released_json_bytes']+=released['json_bytes']
        return released

    def release_payloads(self, keep_paths=()):
        return self.release(set(self.hashes)-set(keep_paths))

    def check_path(self,path):
        self._check_owner(); self.metrics['source_stat_calls']+=1
        require(file_fingerprint(self.resolve(path))==self.marks[path], 'compact source changed; fresh admission required')

    def check(self):
        self._check_owner(); self.metrics['lifecycle_check_calls']+=1
        require(not self.closed,'compact store is closed')
        for path in self.marks: self.check_path(path)
        if self.check_hook is not None: self.check_hook()

    def close(self):
        self._check_owner()
        require(self.borrowers==0,'compact backing still borrowed')
        self.arrays.clear(); self.json.clear(); self.charges.clear()
        self.hashes.clear(); self.resident_bytes=0; self.closed=True

    def __enter__(self): require(not self.closed,'compact store is closed'); return self
    def __exit__(self,*args): self.close()


def _stat(value): return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _size(value, *, maximum=None,retained=0):
    # One ingress/projection graph measurement, never per-row retained rescans.
    from .stock_matrix_reader import _resident_size,_bounded_resident_size
    try:
        if maximum is None: return _resident_size(value)
        return _bounded_resident_size(value,maximum=maximum,retained=lambda:retained)
    finally:
        value=None


def _view_data(handle, *, check=True):
    require(type(handle) is StockFeatureView and handle in _VIEWS,'owner-loaded compact Feature handle required')
    value=_VIEWS[handle]; require(not value['closed'],'compact Feature handle is closed')
    require(value['owner_pid']==os.getpid(),'compact Feature handle belongs to another process')
    if check: value['store'].check()
    return value


def model_feature_binding(feature, selection):
    """Bind an ordered model subset to the unchanged admitted Feature table."""
    value=_view_data(feature)
    return _model_feature_binding(value,selection)


def _model_feature_binding(value, selection):
    if selection is None: return None
    require(type(selection) is list and bool(selection),'nonempty ordered model_feature_selection required')
    definition=value['definition']; spec=definition['spec']; original=spec['feature_selection']
    require([s['id'] for s in original]==spec['ordered_features'],'Feature selection/schema binding mismatch')
    available={s['id']:s for s in original}; selected=[]; seen=set()
    for item in selection:
        fields(item,{'id','semantic_version'},'exact model feature id/version required')
        require(type(item['id']) is str and type(item['semantic_version']) is str and
                item['id'] in available and item==available[item['id']], 'unknown model feature or version')
        require(item['id'] not in seen,'duplicate model feature')
        seen.add(item['id']); selected.append(deepcopy(item))
    schema={c['name']:c for c in definition['schema']}
    return seal({'contract_version':'stock_model_feature_selection_v1',
        'feature_view_ref':definition['feature_view_ref'],
        'historical_feature_inputs_ref':definition['historical_feature_inputs_ref'],
        'selection':selected,'ordered_features':[s['id'] for s in selected],
        'schema':[deepcopy(schema[s['id']]) for s in selected],
        'eligibility':'original_membership_selected_validity_max_known_availability_fit_cutoff_v1'},'model_feature_selection_ref')


def model_common(spec, binding):
    common={k:deepcopy(spec[k]) for k in ('scope','snapshot','pit_policy','calendar','universe',
        'catalog_ref','feature_selection','ordered_features')}
    if binding is not None:
        common.update(feature_selection=deepcopy(binding['selection']),
            ordered_features=list(binding['ordered_features']),model_feature_selection=deepcopy(binding))
    return common


def model_eligibility_ref(binding):
    if binding is None: return None
    return digest({'feature_view_ref':binding['feature_view_ref'],
        'selection':sorted(binding['selection'],key=lambda s:s['id']),
        'schema':sorted(binding['schema'],key=lambda c:c['name']),
        'eligibility':binding['eligibility']})


class StockFeatureView:
    __slots__=('__weakref__',)
    def __init__(self,token,value):
        require(token is _TOKEN,'Feature view requires owner loader')
        value['owner_pid']=os.getpid(); _VIEWS[self]=value
    @property
    def identity(self): return _view_data(self)['definition']['feature_view_ref']
    @property
    def path(self): return _view_data(self)['path']
    @property
    def metrics(self): return deepcopy(_view_data(self)['store'].metrics)
    def to_dict(self): return deepcopy(_view_data(self)['definition'])
    def close(self):
        value=_view_data(self,check=False)
        for state in list(value['prepared'].values()): state.close()
        value['prepared'].clear()
        require(value['borrowers']==0,'compact Feature view still borrowed')
        value['blocks'].clear(); value['store'].close(); value['closed']=True
    def __enter__(self): _view_data(self); return self
    def __exit__(self,*args): self.close()


def load_stock_feature_view(path, *, limits=None, residency="eager",model_feature_selection=None):
    """Admit table files; an explicit subset narrows eager buffer admission.

    The Feature identity and schema remain original. Pass the selection again
    to selected row/matrix consumers; omission keeps full-column semantics.
    """
    require(residency in ("eager","sequential"),"unknown Feature residency mode")
    begin=time.perf_counter(); path=Path(path).absolute(); store=OwnedStore(limits)
    try:
        index_path=str(path/'index.json'); index=store.read_json({'path':index_path},legacy=True,keep=False)
        require(index.get('status')=='COMPLETE' and index.get('contract_version') in
                ('stock_feature_inputs_v2','stock_feature_inputs_v3'),'saved typed Feature input required')
        spec=deepcopy(index['definition']['spec']); days=ordered(spec['feature_sessions'],'Feature sessions')
        securities=ordered(spec['universe'],'Feature securities'); columns=spec['ordered_features']
        column_positions={column:i for i,column in enumerate(columns)}
        require(type(spec['snapshot']) is str and spec['snapshot'] not in ('','current','latest') and
                type(spec['pit_policy']) is str and bool(spec['pit_policy']) and reference(spec['catalog_ref']) and
                reference(index['feature_inputs_ref']),'fixed Feature Snapshot/PIT/catalog/history refs required')
        require(type(columns) is list and bool(columns) and len(set(columns))==len(columns),'Feature column order required')
        ordered(spec['calendar'],'Feature calendar'); ordered(spec['read_sessions'],'Feature read sessions')
        for day in spec['calendar']: _session(day)
        require(set(spec['read_sessions'])<=set(spec['calendar']) and set(days)<=set(spec['read_sessions']),
                'Feature read/date scope mismatch')
        for day,cutoff in spec['cutoff_by_session'].items():
            require(_instant(cutoff).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()==day,
                    'Feature original cutoff must belong to its session')
        require(set(days)<=set(spec['calendar']) and set(spec['cutoff_by_session'])==set(spec['read_sessions']),
                'Feature calendar/cutoff scope mismatch')
        from .stock_matrix_reader import _schema
        schema=index['schema']; _schema(schema,columns)
        owned_path(index['row_index'],path)
        row_index=store.read_json(index['row_index'],key='row_index_ref')
        require(row_index['sessions']==days and row_index['security_ids']==securities and
                row_index['row_count']==len(days)*len(securities) and row_index['order']=='session_security',
                'Feature key index mismatch')
        parts=index['partitions']; groups={}; coverage={c:[] for c in columns}; consumed=[]
        for part in parts:
            require(part['table']=='features' and part['row_index_ref']==row_index['row_index_ref'] and
                    part['schema_digest']==digest(schema),'Feature partition binding mismatch')
            sealed(part,'partition_ref'); start,count=part['row_offset'],part['row_count']
            require(type(start) is int and type(count) is int and start>=0 and count>0 and
                    start%len(securities)==0 and count%len(securities)==0 and start+count<=row_index['row_count'],
                    'complete Feature day partition required')
            owned_path(part['metadata'],path)
            for descriptor in part['buffers'].values(): owned_path(descriptor,path)
            chosen=part['columns']; require(bool(chosen) and len(set(chosen))==len(chosen) and
                                           chosen==[c for c in columns if c in chosen],
                                           'Feature partition columns mismatch')
            expected_types={'values':'float64_le','value_validity':'bool_u8',
                            'available_at_utc_us':'int64_le','available_at_validity':'bool_u8'}
            require(set(part['buffers'])==set(expected_types) and
                    all(v['dtype']==expected_types[k] for k,v in part['buffers'].items()),'Feature physical dtypes mismatch')
            for descriptor in part['buffers'].values():
                fields(descriptor,BUFFER_FIELDS,'exact compact buffer descriptor required')
                require(descriptor['shape']==[count,len(chosen)] and
                        descriptor['file_digest']==descriptor['buffer_digest'],
                        'Feature buffer shapes/digest mismatch')
            for c in chosen: coverage[c].append((start,start+count))
            groups.setdefault((start,count,part['metadata']['path']),[]).append(part)
            consumed.append(deepcopy(part))
        for spans in coverage.values():
            end=0
            for start,stop in sorted(spans): require(start==end,'Feature coverage gap/overlap'); end=stop
            require(end==row_index['row_count'],'Feature table incomplete')
        blocks=[]; parents={}; day_blocks=[None]*len(days); day_admissions=[[] for _ in days]
        for (start,count,_),group in sorted(groups.items()):
            ordinal=len(blocks)
            blocks.append({'start':start,'count':count,'descriptors':group,'parts':None,'rows':None,
                'complete_columns':{c for part in group for c in part['columns']}==set(columns),
                'charge':0,'eligibility':None})
            for day in range(start//len(securities),(start+count)//len(securities)):
                day_admissions[day].append(ordinal)
                if day_blocks[day] is None: day_blocks[day]=ordinal
        require(all(i is not None for i in day_blocks),'Feature block index incomplete')
        definition={'contract_version':'stock_feature_table_view_v1','source_index':{
            'path':index_path,'file_digest':store.hashes[index_path]},'spec':spec,'schema':schema,
            'row_index':deepcopy(index['row_index']),'partitions':consumed,
            'historical_feature_inputs_ref':index['feature_inputs_ref'],
            'historical_content_digest':index.get('content_digest')}
        require(definition['historical_content_digest'] is None or reference(definition['historical_content_digest']),
                'Feature historical content digest must be an explicit ref')
        definition['feature_view_ref']=digest(definition)
        control=_size([definition,parents,row_index,blocks,day_blocks,day_admissions,column_positions],maximum=store.maximum_matrix_bytes,
                      retained=store.resident_bytes)
        store.reserve(control); store.resident_bytes+=control
        store.metrics.update(initialization_seconds=time.perf_counter()-begin,common_key_index_builds=1,
                             resident_bytes=store.resident_bytes,legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        store.check()
        handle=StockFeatureView(_TOKEN,{'path':path,'definition':definition,'store':store,'blocks':blocks,
            'parents':parents,'row_index':row_index,'column_positions':column_positions,'day_blocks':day_blocks,
            'day_admissions':day_admissions,
            'control_paths':{index_path,index['row_index']['path']},'residency':residency,
            'closed':False,'borrowers':0,'prepared':{}})
        binding=_model_feature_binding(_view_data(handle,check=False),model_feature_selection)
        if residency=='eager':
            value=_view_data(handle,check=False)
            for block in blocks:
                if binding is None: _admit_feature_block(value,block)
                else: _admit_model_columns(value,block,binding)
            require(set(parents)==set(days),'Feature parent date coverage mismatch')
        store.metrics['initialization_seconds']=time.perf_counter()-begin
        store.metrics['resident_bytes']=store.resident_bytes
        return handle
    except BaseException:
        store.close(); raise


def _validate_feature_cells(value, group, rows):
    """Validate every cell with block-local clock reuse and part temporaries."""
    import numpy as np
    store=value['store']; count=len(rows)
    largest=max(len(part['columns']) for part, _ in group)
    # K vector, part masks/clocks, predicate temporaries and position headers.
    # Cache growth is reserved separately before parsing/inserting each key.
    workspace=count*32+largest*256+count*largest*16+4096
    cache={}; cache_bytes=0
    cutoffs=meta_flags=meta_present=meta_clock=arrays=part=row=None
    flag=at=positions=None

    def parsed(text):
        nonlocal cache_bytes
        try:
            require(type(text) is str,'Feature clock must be an exact aware string')
            result=cache.get(text)
            if result is not None:
                store.metrics['feature_clock_cache_hits']=store.metrics.get('feature_clock_cache_hits',0)+1
                return result
            if len(cache)==_ADMISSION_CLOCK_CACHE_ENTRIES:
                cache.clear(); cache_bytes=0
            charge=sys.getsizeof(text)+256  # key/int plus conservative map growth
            store.reserve(workspace+cache_bytes+charge)
            result=instant_us(text)  # Cache only a successfully parsed aware clock.
            cache[text]=result; cache_bytes+=charge
            store.metrics['feature_clock_parse_calls']=store.metrics.get('feature_clock_parse_calls',0)+1
            store.metrics['feature_clock_cache_max_entries']=max(
                store.metrics.get('feature_clock_cache_max_entries',0),len(cache))
            return result
        finally:
            text=None

    try:
        store.reserve(workspace)
        cutoffs=np.empty(count,dtype='<i8')
        for i,row in enumerate(rows): cutoffs[i]=parsed(row['knowledge_cutoff'])
        for part,arrays in group:
            positions=[value['column_positions'][column] for column in part['columns']]
            shape=(count,len(positions))
            meta_flags=np.empty(shape,dtype='?'); meta_present=np.empty(shape,dtype='?')
            meta_clock=np.empty(shape,dtype='<i8')
            for i,row in enumerate(rows):
                for j,k in enumerate(positions):
                    flag=row['validity'][k]; at=row['availability'][k]
                    require(type(flag) is bool,'Feature buffer/metadata clock or validity mismatch')
                    meta_flags[i,j]=flag; meta_present[i,j]=at is not None
                    meta_clock[i,j]=0 if at is None else parsed(at)
            require(bool((meta_flags==arrays['value_validity']).all()) and
                    bool((meta_present==arrays['available_at_validity']).all()) and
                    bool(((~meta_present)|(meta_clock==arrays['available_at_utc_us'])).all()),
                    'Feature buffer/metadata clock or validity mismatch')
            require(bool(np.isfinite(arrays['values']).all()),'Feature physical value must be finite')
            require(bool(((arrays['value_validity']!=0)|(arrays['values']==0.0)).all()),
                    'Feature null physical value must be zero')
            require(bool(((arrays['available_at_validity']!=0)|(arrays['available_at_utc_us']==0)).all()),
                    'Feature null physical clock must be zero')
            require(bool(((~meta_present)|(arrays['available_at_utc_us']<=cutoffs[:,None])).all()),
                    'Feature field exceeds its original cutoff')
            store.metrics['feature_validation_array_calls']=store.metrics.get('feature_validation_array_calls',0)+1
            meta_flags=meta_present=meta_clock=None
    finally:
        # Even retained validation tracebacks must release cache/work arrays
        # before the caller credits released Feature buffers.
        cache.clear(); cache=None
        cutoffs=meta_flags=meta_present=meta_clock=arrays=part=row=None
        flag=at=positions=None
        value=group=rows=store=None


def _admit_feature_block(value, block):
    if block['parts'] is not None: return
    import numpy as np
    from .stock_matrix_reader import _CompactRows
    store=value['store']; spec=value['definition']['spec']; columns=spec['ordered_features']
    days=spec['feature_sessions']; securities=spec['universe']; start,count=block['start'],block['count']
    group=[]; rows=metadata=compact=arrays=facts=flags=maximum=None
    try:
        for part in block['descriptors']:
            arrays={k:store.buffer(v) for k,v in part['buffers'].items()}
            require(all(a.shape==(count,len(part['columns'])) for a in arrays.values()),'Feature buffer shapes mismatch')
            group.append((part,arrays)); arrays=None
        metadata=store.read_json(group[0][0]['metadata'],legacy=True,keep=False)
        rows=metadata['rows']; require(len(rows)==count,'Feature metadata row count mismatch')
        for i,row in enumerate(rows):
            off=start+i; day=days[off//len(securities)]
            require((row['security_id'],row['session'])==(securities[off%len(securities)],day) and
                    row['knowledge_cutoff']==spec['cutoff_by_session'][day] and type(row['member']) is bool,
                    'Feature row key/cutoff/member mismatch')
            require(len(row['validity'])==len(row['availability'])==len(row['reasons'])==len(columns),
                    'Feature row metadata width mismatch')
        _validate_feature_cells(value,group,rows)
        compact=_CompactRows(rows,len(columns),np)
        # Reduction happens once per admitted block, never one NumPy temporary
        # per eligibility row. Physical finiteness was checked above.
        store.reserve(compact.bytes+count*32)
        flags=np.full(count,block['complete_columns'],dtype='?')
        maximum=np.full(count,np.iinfo(np.int64).min,dtype='<i8')
        for part,arrays in group:
            flags &= np.all(arrays['value_validity'],axis=1)
            np.maximum(maximum,np.max(arrays['available_at_utc_us'],axis=1,
                where=arrays['available_at_validity'].view('?'),initial=np.iinfo(np.int64).min),out=maximum)
            store.metrics['eligibility_reduction_calls']=store.metrics.get('eligibility_reduction_calls',0)+2
        flags.flags.writeable=False; maximum.flags.writeable=False
        facts=(flags,maximum)
        require(set(metadata['row_references'])==set(days[start//len(securities):(start+count)//len(securities)]),
                'Feature metadata parent date scope mismatch')
        new_parents={}
        for day,parent in metadata['row_references'].items():
            require(reference(parent.get('feature_ref')) and reference(parent.get('qlib_view_ref')),
                    'explicit Feature parent refs required')
            require(day not in value['parents'] or value['parents'][day]==parent,'conflicting Feature parent reference')
            if day not in value['parents']: new_parents[day]=deepcopy(parent)
        parent_charge=_size(new_parents,maximum=store.maximum_matrix_bytes,
            retained=store.resident_bytes+compact.bytes+flags.nbytes+maximum.nbytes)
        charge=compact.bytes+flags.nbytes+maximum.nbytes
        store.reserve(charge+parent_charge); store.resident_bytes+=charge+parent_charge
        value['parents'].update(new_parents)
        block.update(parts=group,rows=compact,eligibility=facts,charge=charge)
        store.metrics['feature_block_admissions']=store.metrics.get('feature_block_admissions',0)+1
        store.metrics['resident_bytes']=store.resident_bytes
    except BaseException:
        # Tracebacks must not keep views alive after an owner credits release.
        group.clear(); rows=metadata=compact=arrays=facts=flags=maximum=None
        raise


def _admit_model_columns(value, block, binding):
    """Admit selected physical parts into the existing owner's byte/JSON caches."""
    import numpy as np
    from .stock_matrix_reader import _CompactRows
    store=value['store']; spec=value['definition']['spec']; columns=binding['ordered_features']
    selected=set(columns); count=block['count']; start=block['start']; universe=spec['universe']
    require(selected <= {c for p in block['descriptors'] for c in p['columns']},
            'selected Feature block is missing columns')
    if 'model_parts' not in block:
        control=4096+16*len(spec['ordered_features'])
        store.reserve(control); store.resident_bytes+=control
        block.update(model_parts={},model_charge=control)
    cache=block['model_parts']
    new=[]; metadata=None
    try:
        if block['parts'] is not None:
            for part,arrays in block['parts']:
                if selected.intersection(part['columns']) and part['partition_ref'] not in cache:
                    store.reserve(512); store.resident_bytes+=512; block['model_charge']+=512
                    cache[part['partition_ref']]=(part,arrays)
        for part in block['descriptors']:
            if not selected.intersection(part['columns']) or part['partition_ref'] in cache: continue
            store.reserve((len(new)+1)*1024)
            arrays={k:store.buffer(v) for k,v in part['buffers'].items()}
            require(all(a.shape==(count,len(part['columns'])) for a in arrays.values()),'Feature buffer shapes mismatch')
            new.append((part,arrays))
        if new or 'model_rows' not in block:
            if block['parts'] is not None and 'model_rows' not in block:
                block['model_rows']=block['rows']
                store.metrics['active_model_feature_blocks']=store.metrics.get('active_model_feature_blocks',0)+1
            else:
                metadata=store.read_json(block['descriptors'][0]['metadata'],legacy=True)
                rows=metadata['rows']; require(len(rows)==count,'Feature metadata row count mismatch')
                for i,row in enumerate(rows):
                    off=start+i; day=spec['feature_sessions'][off//len(universe)]
                    require((row['security_id'],row['session'])==(universe[off%len(universe)],day) and
                            row['knowledge_cutoff']==spec['cutoff_by_session'][day] and type(row['member']) is bool,
                            'Feature row key/cutoff/member mismatch')
                    require(len(row['validity'])==len(row['availability'])==len(row['reasons'])==len(spec['ordered_features']),
                            'Feature row metadata width mismatch')
                if new: _validate_feature_cells(value,new,rows)
                if 'model_rows' not in block:
                    store.reserve(_size(rows,maximum=store.maximum_matrix_bytes,
                        retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)+count*256+4096)
                    compact=_CompactRows(rows,len(spec['ordered_features']),np)
                    parents={}
                    require(set(metadata['row_references'])==set(spec['feature_sessions'][
                        start//len(universe):(start+count)//len(universe)]),'Feature metadata parent date scope mismatch')
                    for day,parent in metadata['row_references'].items():
                        require(reference(parent.get('feature_ref')) and reference(parent.get('qlib_view_ref')),
                                'explicit Feature parent refs required')
                        require(day not in value['parents'] or value['parents'][day]==parent,'conflicting Feature parent reference')
                        if day not in value['parents']: parents[day]=deepcopy(parent)
                    charge=compact.bytes+_size(parents)
                    store.reserve(charge); store.resident_bytes+=charge
                    block['model_rows']=compact; block['model_charge']+=compact.bytes
                    store.metrics['active_model_feature_blocks']=store.metrics.get('active_model_feature_blocks',0)+1
                    value['parents'].update(parents)
        for part,arrays in new:
            store.reserve(1024); store.resident_bytes+=1024
            block['model_charge']+=1024
            cache[part['partition_ref']]=(part,arrays)
            store.metrics['model_column_part_admissions']=store.metrics.get('model_column_part_admissions',0)+1
        key=tuple(sorted(selected))
        if block.get('model_eligibility_key')!=key:
            store.reserve(count*18+count*32+4096)
            flags=np.ones(count,dtype='?'); maximum=np.full(count,np.iinfo(np.int64).min,dtype='<i8')
            for part,arrays in cache.values():
                positions=[j for j,c in enumerate(part['columns']) if c in selected]
                if not positions: continue
                store.reserve(count*18+count*len(positions)*17+4096)
                flags &= np.all(arrays['value_validity'][:,positions],axis=1)
                np.maximum(maximum,np.max(arrays['available_at_utc_us'][:,positions],axis=1,
                    where=arrays['available_at_validity'][:,positions].view('?'),
                    initial=np.iinfo(np.int64).min),out=maximum)
            flags.flags.writeable=maximum.flags.writeable=False
            old=block.get('model_eligibility')
            if old is None:
                charge=flags.nbytes+maximum.nbytes
                store.resident_bytes+=charge; block['model_charge']+=charge
            block.update(model_eligibility=(flags,maximum),model_eligibility_key=key)
        store.metrics['resident_bytes']=store.resident_bytes
        return cache,block['model_rows'],block['model_eligibility']
    except BaseException:
        new.clear(); cache=arrays=rows=metadata=compact=flags=maximum=old=None
        raise


def set_feature_window(handle, offsets, *, model_feature_selection=None):
    """Admit the complete selected day blocks; eager owners keep their choice."""
    value=_view_data(handle); store=value['store']
    binding=_model_feature_binding(value,model_feature_selection)
    if value['residency']=='eager' and binding is None: return
    require(store.borrowers==0 and value.get('active',0)==0,'Feature window still borrowed')
    width=len(value['definition']['spec']['universe']); wanted=set()
    for off in offsets:
        require(type(off) is int and 0<=off<value['row_index']['row_count'],'Feature offset outside view')
        wanted.update(value['day_admissions'][off//width])
    if value['residency']=='sequential': _trim_feature_window(value,wanted)
    try:
        for ordinal in sorted(wanted):
            if binding is None: _admit_feature_block(value,value['blocks'][ordinal])
            else: _admit_model_columns(value,value['blocks'][ordinal],binding)
    except BaseException:
        # Skip lifecycle stat checks during cleanup of a failed admission.
        _trim_feature_window(value,set())
        raise
    store.metrics['active_feature_blocks']=sum(b['parts'] is not None for b in value['blocks'])
    store.metrics['resident_bytes']=store.resident_bytes
    store.check()


def clear_feature_window(handle):
    """Failure cleanup drops owned leaves without revisiting damaged files."""
    value=_view_data(handle,check=False)
    if value['residency']=='sequential':
        require(value['store'].borrowers==0 and value.get('active',0)==0,'Feature window still borrowed')
        _trim_feature_window(value,set())


def _trim_feature_window(value, wanted):
    store=value['store']; keep=set(value['control_paths'])
    for ordinal,block in enumerate(value['blocks']):
        if ordinal not in wanted and 'training_block_proofs' in block:
            store.resident_bytes-=block.pop('training_block_proof_charge',0)
            block.pop('training_block_proofs',None)
        if ordinal in wanted:
            for part in block['descriptors']:
                keep.add(part['metadata']['path']); keep.update(v['path'] for v in part['buffers'].values())
        elif block['parts'] is not None:
            charge=block['charge']; block.update(parts=None,rows=None,eligibility=None,charge=0)
            store.resident_bytes-=charge
            store.metrics['released_feature_metadata_bytes']=store.metrics.get('released_feature_metadata_bytes',0)+charge
        if ordinal not in wanted and 'model_parts' in block:
            charge=block.pop('model_charge',0); store.resident_bytes-=charge
            if 'model_rows' in block: store.metrics['active_model_feature_blocks']-=1
            for key in ('model_rows','model_parts','model_eligibility','model_eligibility_key'): block.pop(key,None)
    store.release_payloads(keep)


def feature_training_blocks(handle, offsets, *, model_feature_selection=None):
    """Bind selected native columns once per live immutable Feature block.

    The small proofs live in the existing Feature window and are charged to its
    store. No second table, value panel or process-global digest cache exists.
    """
    try:
        from hashlib import sha256
        value=_view_data(handle); store=value['store']; spec=value['definition']['spec']
        binding=model_feature_binding(handle,model_feature_selection)
        columns=spec['ordered_features'] if binding is None else binding['ordered_features']
        require(all(type(off) is int and 0<=off<value['row_index']['row_count'] for off in offsets),
                'training proof offsets exceed the Feature grid')
        width=len(spec['universe']); wanted=sorted({value['day_blocks'][off//width] for off in offsets})
        result=[]; key=tuple(columns); copies=0
        for ordinal in wanted:
            block=value['blocks'][ordinal]
            cached=block.get('training_block_proofs',{}).get(key)
            if cached is None:
                if binding is None:
                    _admit_feature_block(value,block); parts=block['parts']; rows=block['rows']
                else:
                    parts,rows,_=_admit_model_columns(value,block,binding); parts=parts.values()
                fields_by_name={}; source_parts=[]
                for part,arrays in parts:
                    chosen=[(i,name) for i,name in enumerate(part['columns']) if name in key]
                    if not chosen: continue
                    source_parts.append({'partition_ref':part['partition_ref'],
                        'metadata':deepcopy(part['metadata']),'buffers':deepcopy(part['buffers'])})
                    for i,name in chosen:
                        refs={}
                        for buffer_name,a in arrays.items():
                            # A single strided column copy is bounded before it is
                            # hashed, never a full fit-window text projection.
                            store.reserve(block['count']*a.dtype.itemsize+4096)
                            payload=a[:,i].tobytes(order='C')
                            refs[buffer_name]={'dtype':part['buffers'][buffer_name]['dtype'],
                                'shape':[block['count']], 'buffer_digest':'sha256:'+sha256(payload).hexdigest()}
                            payload=None
                        fields_by_name[name]=refs
                store.reserve(block['count']*8+4096)
                # Knowledge codes are local interning slots. Bind their actual
                # clocks rather than unrelated dictionary/code assignments.
                knowledge=digest_array_rows(rows.dictionary[int(code)] for code in rows.knowledge)
                members='sha256:'+sha256(memoryview(rows.member)).hexdigest()
                dependency={'contract_version':'stock_feature_training_block_dependency_v1',
                    'row_index_ref':value['row_index']['row_index_ref'],'first_row':block['start'],
                    'row_count':block['count'],'ordered_features':list(columns),
                    'columns':{name:fields_by_name[name] for name in columns},
                    'member_ref':members,'knowledge_ref':knowledge}
                cached=seal({'contract_version':'stock_feature_training_block_v1',
                    **{k:v for k,v in dependency.items() if k!='contract_version'},
                    'dependency_ref':digest(dependency),'source_parts':source_parts},'feature_block_ref')
                charge=_size(cached,maximum=store.maximum_matrix_bytes,
                    retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)+1024
                store.reserve(charge); store.resident_bytes+=charge
                block.setdefault('training_block_proofs',{})[key]=cached
                block['training_block_proof_charge']=block.get('training_block_proof_charge',0)+charge
                store.metrics['training_block_proof_builds']=store.metrics.get('training_block_proof_builds',0)+1
            sealed(cached,'feature_block_ref')
            copies+=_size(cached)+1024;store.reserve(copies)
            result.append(deepcopy(cached))
        store.check()
        return result

    finally:
        value=store=spec=binding=columns=block=cached=parts=rows=part=arrays=a=payload=result=handle=None


def _offset_runs(value, offsets, *, chunk=1024):
    """Preserve supplied row order with O(1) day index and one block run."""
    width=len(value['definition']['spec']['universe']); ordinal=None; local=[]; first=0; total=0
    for off in offsets:
        require(type(off) is int and 0<=off<value['row_index']['row_count'],'Feature offset outside view')
        chosen=value['day_blocks'][off//width]
        if local and (chosen!=ordinal or len(local)==chunk):
            value['store'].metrics['block_run_visits']=value['store'].metrics.get('block_run_visits',0)+1
            yield first,value['blocks'][ordinal],local
            first=total; local=[]
        ordinal=chosen; local.append(off-value['blocks'][chosen]['start']); total+=1
    if local:
        value['store'].metrics['block_run_visits']=value['store'].metrics.get('block_run_visits',0)+1
        yield first,value['blocks'][ordinal],local


def iter_feature_rows(handle, offsets, *, model_feature_selection=None):
    value=_view_data(handle); spec=value['definition']['spec']; width=len(spec['universe'])
    binding=_model_feature_binding(value,model_feature_selection)
    columns=spec['ordered_features'] if binding is None else binding['ordered_features']
    positions={c:i for i,c in enumerate(columns)}
    source_positions=[value['column_positions'][c] for c in columns]
    for _,block,locals_ in _offset_runs(value,offsets):
        if binding is None:
            _admit_feature_block(value,block); parts=block['parts']; compact=block['rows']
        else:
            cache,compact,_=_admit_model_columns(value,block,binding); parts=cache.values()
        for local in locals_:
            off=block['start']+local
            values=[None]*len(columns); flags=[False]*len(values)
            for part,arrays in parts:
                for j,c in enumerate(part['columns']):
                    if c not in positions: continue
                    k=positions[c]; flag=bool(arrays['value_validity'][local,j]); flags[k]=flag
                    values[k]=float(arrays['values'][local,j]) if flag else None
            if binding is None:
                row=compact.row(local,security=spec['universe'][off%width],
                    session=spec['feature_sessions'][off//width],values=values,validity=flags)
            else:
                row={'security_id':spec['universe'][off%width],'session':spec['feature_sessions'][off//width],
                    'member':bool(compact.member[local]),'values':values,'validity':flags,
                    'knowledge_cutoff':deepcopy(compact.dictionary[int(compact.knowledge[local])]),
                    'source_refs':deepcopy(compact.dictionary[int(compact.sources[local])])}
                for name in ('reasons','availability'):
                    original=compact.dictionary[int(getattr(compact,name)[local])]
                    row[name]=deepcopy([original[j] for j in source_positions])
            yield row
    value['store'].check()


def feature_rows(handle, offsets, *, model_feature_selection=None):
    return list(iter_feature_rows(handle,offsets,model_feature_selection=model_feature_selection))


def iter_feature_eligibility(handle, offsets, *, model_feature_selection=None):
    """Reuse admitted block reductions without per-row NumPy workspaces."""
    from .stock_label_contracts import _FeatureEligibility
    value=_view_data(handle); minimum=-(2**63)
    binding=_model_feature_binding(value,model_feature_selection)
    for _,block,locals_ in _offset_runs(value,offsets):
        if binding is None:
            _admit_feature_block(value,block); compact=block['rows']; flags,maximum=block['eligibility']
        else:
            _,compact,(flags,maximum)=_admit_model_columns(value,block,binding)
        for local in locals_:
            at=int(maximum[local]); complete=bool(flags[local])
            yield _FeatureEligibility(bool(compact.member[local]),complete,complete,
                compact.dictionary[int(compact.knowledge[local])],None if at==minimum else at)
    value['store'].check()


def training_matrix(handle,offsets,cutoff,*,model_feature_selection=None):
    """Gather X in block batches, preserving the caller's exact row order."""
    import numpy as np
    value=_view_data(handle); spec=value['definition']['spec']; instant=instant_us(cutoff)
    binding=_model_feature_binding(value,model_feature_selection)
    columns=spec['ordered_features'] if binding is None else binding['ordered_features']
    column_positions={c:i for i,c in enumerate(columns)}
    require(all(type(off) is int and 0<=off<value['row_index']['row_count'] for off in offsets),
            'Feature matrix offsets outside view')
    # One check per selected block before any output allocation. The day index
    # preserves the original first matching metadata block semantics.
    width=len(spec['universe']); chosen={value['day_blocks'][off//width] for off in offsets}
    require(all(value['blocks'][i]['complete_columns'] for i in chosen),
            'selected training Feature block is missing columns')
    for i in sorted(chosen):
        if binding is None: _admit_feature_block(value,value['blocks'][i])
        else: _admit_model_columns(value,value['blocks'][i],binding)
    matrix=np.empty((len(offsets),len(columns)),dtype='<f8')
    try:
        for first,block,locals_ in _offset_runs(value,offsets):
            if binding is None: parts=block['parts']; rows=block['rows']; flags,maximum=block['eligibility']
            else:
                cache,rows,(flags,maximum)=_admit_model_columns(value,block,binding); parts=cache.values()
            indexes=np.asarray(locals_,dtype=np.intp)
            require(bool(rows.member[indexes].all()) and bool(flags[indexes].all()) and
                    bool(((maximum[indexes]!=-(2**63)) & (maximum[indexes]<=instant)).all()) and
                    all(instant_us(rows.dictionary[int(code)])<=instant for code in set(rows.knowledge[indexes])),
                    'selected training Feature membership/clock mismatch')
            # Admission validated every individual availability <= original K;
            # max availability and K above retain the complete fit cutoff check.
            for part,arrays in parts:
                selected=[j for j,c in enumerate(part['columns']) if c in column_positions]
                if not selected: continue
                positions=[column_positions[part['columns'][j]] for j in selected]
                workspace=len(indexes)*len(positions)*8+len(indexes)*32
                value['store'].reserve(matrix.nbytes+workspace)
                matrix[first:first+len(indexes),positions]=(arrays['values'][indexes] if binding is None
                    else arrays['values'][indexes[:,None],selected])
                value['store'].metrics['training_block_gathers']=value['store'].metrics.get('training_block_gathers',0)+1
        require(bool(np.isfinite(matrix).all()),'selected training Feature values must be finite')
        matrix.flags.writeable=False; value['store'].check(); return matrix
    except BaseException:
        matrix=indexes=arrays=rows=flags=maximum=block=cache=parts=None
        raise

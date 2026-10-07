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
import time

from .stock_artifacts import digest
from .stock_fold_inputs import require, ordered, file_fingerprint
from .stock_label_contracts import _instant, _session
from .stock_matrix_storage import DTYPES, BUFFER_FIELDS, instant_us

DEFAULT_LIMITS = {'maximum_source_bytes':8*1024**3,
                  'maximum_matrix_bytes':512*1024**2,
                  'maximum_parent_bytes':64*1024**2}
_TOKEN = object()
_VIEWS = WeakKeyDictionary()


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
    require(reference(value.get(key)) and value[key]==digest({k:v for k,v in value.items() if k!=key}),
            'compact '+key+' identity mismatch')


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
        with physical.open('rb') as stream:
            require(_stat(os.fstat(stream.fileno()))==mark,'compact source changed before read')
            payload=stream.read(mark[2]+1)
            require(len(payload)==mark[2] and _stat(os.fstat(stream.fileno()))==mark and
                    file_fingerprint(physical)==mark,'compact source changed during read')
            self.metrics['source_stat_calls']+=1
        actual='sha256:'+sha256(payload).hexdigest()
        require(expected is None or actual==expected,'compact file digest mismatch')
        self.marks[path]=mark; self.hashes[path]=actual
        self.metrics['file_hash_calls']+=1; self.metrics['hash_bytes']+=len(payload)
        self.metrics['source_bytes']+=len(payload)
        return payload

    def read_json(self, descriptor, *, key=None, legacy=False, keep=True):
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

    def buffer(self, descriptor):
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
        try:
            require(not array.flags.writeable,'owned bytes must remain readonly')
            if descriptor['dtype']=='bool_u8': require(bool(((array==0)|(array==1)).all()),'compact bool bytes must be 0/1')
        except BaseException:
            array=payload=None
            raise
        return array

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
    if maximum is None: return _resident_size(value)
    return _bounded_resident_size(value,maximum=maximum,retained=lambda:retained)


def _view_data(handle, *, check=True):
    require(type(handle) is StockFeatureView and handle in _VIEWS,'owner-loaded compact Feature handle required')
    value=_VIEWS[handle]; require(not value['closed'],'compact Feature handle is closed')
    require(value['owner_pid']==os.getpid(),'compact Feature handle belongs to another process')
    if check: value['store'].check()
    return value


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


def load_stock_feature_view(path, *, limits=None, residency="eager"):
    """Ordinary ingress stops at Feature table files, never source ancestors."""
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
        blocks=[]; parents={}; day_blocks=[None]*len(days)
        for (start,count,_),group in sorted(groups.items()):
            ordinal=len(blocks)
            blocks.append({'start':start,'count':count,'descriptors':group,'parts':None,'rows':None,
                'complete_columns':{c for part in group for c in part['columns']}==set(columns),
                'charge':0,'eligibility':None})
            for day in range(start//len(securities),(start+count)//len(securities)):
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
        control=_size([definition,parents,row_index,blocks,day_blocks,column_positions],maximum=store.maximum_matrix_bytes,
                      retained=store.resident_bytes)
        store.reserve(control); store.resident_bytes+=control
        store.metrics.update(initialization_seconds=time.perf_counter()-begin,common_key_index_builds=1,
                             resident_bytes=store.resident_bytes,legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        store.check()
        handle=StockFeatureView(_TOKEN,{'path':path,'definition':definition,'store':store,'blocks':blocks,
            'parents':parents,'row_index':row_index,'column_positions':column_positions,'day_blocks':day_blocks,
            'control_paths':{index_path,index['row_index']['path']},'residency':residency,
            'closed':False,'borrowers':0,'prepared':{}})
        if residency=='eager':
            value=_view_data(handle,check=False)
            for block in blocks: _admit_feature_block(value,block)
            require(set(parents)==set(days),'Feature parent date coverage mismatch')
        store.metrics['initialization_seconds']=time.perf_counter()-begin
        store.metrics['resident_bytes']=store.resident_bytes
        return handle
    except BaseException:
        store.close(); raise


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
            row['values']=[None]*len(columns)
        for part,arrays in group:
            for j,column in enumerate(part['columns']):
                k=value['column_positions'][column]
                for i,row in enumerate(rows):
                    flag=bool(arrays['value_validity'][i,j]); at=row['availability'][k]
                    require(row['validity'][k] is flag and bool(arrays['available_at_validity'][i,j]) is (at is not None) and
                            (at is None or int(arrays['available_at_utc_us'][i,j])==instant_us(at)),
                            'Feature buffer/metadata clock or validity mismatch')
                    cell=float(arrays['values'][i,j]); require(np.isfinite(cell),'Feature physical value must be finite')
                    require(flag or cell==0.0,'Feature null physical value must be zero')
                    require(at is not None or int(arrays['available_at_utc_us'][i,j])==0,
                            'Feature null physical clock must be zero')
                    row['values'][k]=cell if flag else None
                    require(at is None or _instant(at)<=_instant(row['knowledge_cutoff']),
                            'Feature field exceeds its original cutoff')
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


def set_feature_window(handle, offsets):
    """Admit the complete selected day blocks; eager owners keep their choice."""
    value=_view_data(handle); store=value['store']
    if value['residency']=='eager': return
    require(store.borrowers==0 and value.get('active',0)==0,'Feature window still borrowed')
    width=len(value['definition']['spec']['universe']); wanted=set()
    for off in offsets:
        require(type(off) is int and 0<=off<value['row_index']['row_count'],'Feature offset outside view')
        wanted.add(value['day_blocks'][off//width])
    _trim_feature_window(value,wanted)
    try:
        for ordinal in sorted(wanted): _admit_feature_block(value,value['blocks'][ordinal])
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
        if ordinal in wanted:
            for part in block['descriptors']:
                keep.add(part['metadata']['path']); keep.update(v['path'] for v in part['buffers'].values())
        elif block['parts'] is not None:
            charge=block['charge']; block.update(parts=None,rows=None,eligibility=None,charge=0)
            store.resident_bytes-=charge
            store.metrics['released_feature_metadata_bytes']=store.metrics.get('released_feature_metadata_bytes',0)+charge
    store.release_payloads(keep)


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


def iter_feature_rows(handle, offsets):
    value=_view_data(handle); spec=value['definition']['spec']; width=len(spec['universe'])
    for _,block,locals_ in _offset_runs(value,offsets):
        _admit_feature_block(value,block)
        for local in locals_:
            off=block['start']+local
            values=[None]*len(spec['ordered_features']); flags=[False]*len(values)
            for part,arrays in block['parts']:
                for j,c in enumerate(part['columns']):
                    k=value['column_positions'][c]; flag=bool(arrays['value_validity'][local,j]); flags[k]=flag
                    values[k]=float(arrays['values'][local,j]) if flag else None
            yield block['rows'].row(local,security=spec['universe'][off%width],
                session=spec['feature_sessions'][off//width],values=values,validity=flags)
    value['store'].check()


def feature_rows(handle, offsets):
    return list(iter_feature_rows(handle,offsets))


def iter_feature_eligibility(handle, offsets):
    """Reuse admitted block reductions without per-row NumPy workspaces."""
    from .stock_label_contracts import _FeatureEligibility
    value=_view_data(handle); minimum=-(2**63)
    for _,block,locals_ in _offset_runs(value,offsets):
        _admit_feature_block(value,block)
        compact=block['rows']; flags,maximum=block['eligibility']
        for local in locals_:
            at=int(maximum[local]); complete=bool(flags[local])
            yield _FeatureEligibility(bool(compact.member[local]),complete,complete,
                compact.dictionary[int(compact.knowledge[local])],None if at==minimum else at)
    value['store'].check()


def training_matrix(handle,offsets,cutoff):
    """Gather X in block batches, preserving the caller's exact row order."""
    import numpy as np
    value=_view_data(handle); spec=value['definition']['spec']; instant=instant_us(cutoff)
    require(all(type(off) is int and 0<=off<value['row_index']['row_count'] for off in offsets),
            'Feature matrix offsets outside view')
    # One check per selected block before any output allocation. The day index
    # preserves the original first matching metadata block semantics.
    width=len(spec['universe']); chosen={value['day_blocks'][off//width] for off in offsets}
    require(all(value['blocks'][i]['complete_columns'] for i in chosen),
            'selected training Feature block is missing columns')
    for i in sorted(chosen): _admit_feature_block(value,value['blocks'][i])
    matrix=np.empty((len(offsets),len(spec['ordered_features'])),dtype='<f8')
    try:
        for first,block,locals_ in _offset_runs(value,offsets):
            rows=block['rows']; indexes=np.asarray(locals_,dtype=np.intp)
            flags,maximum=block['eligibility']
            require(bool(rows.member[indexes].all()) and bool(flags[indexes].all()) and
                    bool(((maximum[indexes]!=-(2**63)) & (maximum[indexes]<=instant)).all()) and
                    all(instant_us(rows.dictionary[int(code)])<=instant for code in set(rows.knowledge[indexes])),
                    'selected training Feature membership/clock mismatch')
            # Admission validated every individual availability <= original K;
            # max availability and K above retain the complete fit cutoff check.
            for part,arrays in block['parts']:
                positions=[value['column_positions'][column] for column in part['columns']]
                workspace=len(indexes)*len(positions)*8+len(indexes)*32
                value['store'].reserve(matrix.nbytes+workspace)
                matrix[first:first+len(indexes),positions]=arrays['values'][indexes]
                value['store'].metrics['training_block_gathers']=value['store'].metrics.get('training_block_gathers',0)+1
        require(bool(np.isfinite(matrix).all()),'selected training Feature values must be finite')
        matrix.flags.writeable=False; value['store'].check(); return matrix
    except BaseException:
        matrix=indexes=arrays=rows=flags=maximum=block=None
        raise

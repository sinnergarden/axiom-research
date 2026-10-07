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
        self.check_hook=None
        self.metrics={'file_hash_calls':0,'hash_bytes':0,'source_bytes':0,'json_decode_calls':0,
                      'owned_buffer_bytes':0,'mmap_opens':0,'fold_projection_calls':0,'largest_parent_bytes':0,
                      'source_stat_calls':0,'lifecycle_check_calls':0}

    def _check_owner(self):
        require(self.owner_pid==os.getpid(),'compact store belongs to another process')

    @property
    def maximum_matrix_bytes(self): return self.limits['maximum_matrix_bytes']

    def reserve(self, amount):
        self._check_owner()
        require(type(amount) is int and amount>=0 and
                self.shared_bytes+self.resident_bytes+self.lease_bytes+amount<=self.maximum_matrix_bytes,
                'compact resident byte budget exceeded')

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
            require(path in self.arrays or path in self.json,'released ingress cannot be readmitted inside a view')
            return None
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
                self.json[path]=value
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
            self.arrays[path]=payload
        else: self.check_path(path)
        require(self.hashes[path]==descriptor['file_digest'],'conflicting compact buffer digest')
        payload=self.arrays[path]
        require(len(payload)==count*np.dtype(dtype).itemsize,'conflicting compact buffer shape')
        array=np.frombuffer(payload,dtype=dtype).reshape(shape)
        require(not array.flags.writeable,'owned bytes must remain readonly')
        if descriptor['dtype']=='bool_u8': require(bool(((array==0)|(array==1)).all()),'compact bool bytes must be 0/1')
        return array

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
        self.arrays.clear(); self.json.clear(); self.closed=True

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
        value['store'].close(); value['blocks'].clear(); value['closed']=True
    def __enter__(self): _view_data(self); return self
    def __exit__(self,*args): self.close()


def load_stock_feature_view(path, *, limits=None):
    """Ordinary ingress stops at Feature table files, never source ancestors."""
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
            arrays={k:store.buffer(v) for k,v in part['buffers'].items()}
            require(set(arrays)=={'values','value_validity','available_at_utc_us','available_at_validity'} and
                    all(a.shape==(count,len(chosen)) for a in arrays.values()),'Feature buffer shapes mismatch')
            for c in chosen: coverage[c].append((start,start+count))
            groups.setdefault((start,count,part['metadata']['path']),[]).append((part,arrays))
            consumed.append(deepcopy(part))
        for spans in coverage.values():
            end=0
            for start,stop in sorted(spans): require(start==end,'Feature coverage gap/overlap'); end=stop
            require(end==row_index['row_count'],'Feature table incomplete')
        blocks=[]; parents={}
        from .stock_matrix_reader import _CompactRows
        import numpy as np
        for (start,count,_),group in sorted(groups.items()):
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
                    k=column_positions[column]
                    for i,row in enumerate(rows):
                        flag=bool(arrays['value_validity'][i,j]); at=row['availability'][k]
                        require(row['validity'][k] is flag and bool(arrays['available_at_validity'][i,j]) is (at is not None) and
                                (at is None or int(arrays['available_at_utc_us'][i,j])==instant_us(at)),
                                'Feature buffer/metadata clock or validity mismatch')
                        value=float(arrays['values'][i,j]); require(np.isfinite(value),'Feature physical value must be finite')
                        require(flag or value==0.0,'Feature null physical value must be zero')
                        require(at is not None or int(arrays['available_at_utc_us'][i,j])==0,
                                'Feature null physical clock must be zero')
                        row['values'][k]=value if flag else None
                        require(at is None or _instant(at)<=_instant(row['knowledge_cutoff']),
                                'Feature field exceeds its original cutoff')
            compact=_CompactRows(rows,len(columns),np); store.reserve(compact.bytes); store.resident_bytes+=compact.bytes
            complete_columns={c for part,_ in group for c in part['columns']}==set(columns)
            blocks.append({'start':start,'count':count,'parts':group,'rows':compact,
                           'complete_columns':complete_columns})
            require(set(metadata['row_references'])==set(days[start//len(securities):(start+count)//len(securities)]),
                    'Feature metadata parent date scope mismatch')
            for day,parent in metadata['row_references'].items():
                require(day not in parents or parents[day]==parent,'conflicting Feature parent reference')
                parents[day]=deepcopy(parent)
            del rows,metadata
        require(set(parents)==set(days),'Feature parent date coverage mismatch')
        require(all(reference(p.get('feature_ref')) and reference(p.get('qlib_view_ref')) for p in parents.values()),
                'explicit Feature parent refs required')
        definition={'contract_version':'stock_feature_table_view_v1','source_index':{
            'path':index_path,'file_digest':store.hashes[index_path]},'spec':spec,'schema':schema,
            'row_index':deepcopy(index['row_index']),'partitions':consumed,
            'historical_feature_inputs_ref':index['feature_inputs_ref'],
            'historical_content_digest':index.get('content_digest')}
        require(definition['historical_content_digest'] is None or reference(definition['historical_content_digest']),
                'Feature historical content digest must be an explicit ref')
        definition['feature_view_ref']=digest(definition)
        control=_size([definition,parents,row_index],maximum=store.maximum_matrix_bytes,
                      retained=store.resident_bytes)
        store.reserve(control); store.resident_bytes+=control
        store.metrics.update(initialization_seconds=time.perf_counter()-begin,common_key_index_builds=1,
                             resident_bytes=store.resident_bytes,legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        store.check()
        return StockFeatureView(_TOKEN,{'path':path,'definition':definition,'store':store,'blocks':blocks,
            'parents':parents,'row_index':row_index,'column_positions':column_positions,
            'closed':False,'borrowers':0,'prepared':{}})
    except BaseException:
        store.close(); raise


def iter_feature_rows(handle, offsets):
    value=_view_data(handle); spec=value['definition']['spec']; width=len(spec['universe'])
    for off in offsets:
        require(type(off) is int and 0<=off<value['row_index']['row_count'],'Feature offset outside view')
        block=next(b for b in value['blocks'] if b['start']<=off<b['start']+b['count']); local=off-block['start']
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
    """Project checks without materializing values/reasons/clock row panels."""
    from .stock_label_contracts import _FeatureEligibility
    import numpy as np
    value=_view_data(handle)
    for off in offsets:
        require(type(off) is int and 0<=off<value['row_index']['row_count'],'Feature offset outside view')
        block=next(b for b in value['blocks'] if b['start']<=off<b['start']+b['count']); local=off-block['start']
        compact=block['rows']; complete=block['complete_columns']; valid=complete; maximum=None
        for part,arrays in block['parts']:
            flags=bool(arrays['value_validity'][local].all()); valid=valid and flags
            # A masked physical zero is logical None, hence FEATURE_MISSING
            # precedes FEATURE_INVALID exactly as in the original dict route.
            complete=complete and flags and bool(np.isfinite(arrays['values'][local]).all())
            known=arrays['available_at_validity'][local].astype(bool)
            if bool(known.any()):
                at=int(arrays['available_at_utc_us'][local][known].max())
                maximum=at if maximum is None else max(maximum,at)
        yield _FeatureEligibility(bool(compact.member[local]),complete,valid,
            compact.dictionary[int(compact.knowledge[local])],maximum)
    value['store'].check()


def training_matrix(handle,offsets,cutoff):
    """Gather only X from admitted typed bytes, without training row panels."""
    import numpy as np
    value=_view_data(handle); spec=value['definition']['spec']; instant=_instant(cutoff)
    require(all(type(off) is int and 0<=off<value['row_index']['row_count'] for off in offsets),
            'Feature matrix offsets outside view')
    require(all(next(b for b in value['blocks'] if b['start']<=off<b['start']+b['count'])['complete_columns']
                for off in offsets),'selected training Feature block is missing columns')
    matrix=np.empty((len(offsets),len(spec['ordered_features'])),dtype='<f8')
    for i,off in enumerate(offsets):
        block=next(b for b in value['blocks'] if b['start']<=off<b['start']+b['count']); local=off-block['start']; rows=block['rows']
        knowledge=rows.dictionary[int(rows.knowledge[local])]; availability=rows.dictionary[int(rows.availability[local])]
        require(bool(rows.member[local]) and _instant(knowledge)<=instant and any(a is not None for a in availability) and
                all(a is None or _instant(a)<=instant for a in availability),
                'selected training Feature membership/clock mismatch')
        for part,arrays in block['parts']:
            require(bool(arrays['value_validity'][local].all()),'selected training Feature validity mismatch')
            for j,column in enumerate(part['columns']): matrix[i,value['column_positions'][column]]=arrays['values'][local,j]
    require(bool(np.isfinite(matrix).all()),'selected training Feature values must be finite')
    matrix.flags.writeable=False; value['store'].check(); return matrix

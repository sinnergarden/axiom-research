"""Admission and bounded projection of immutable matrix inputs.

This module reads saved bytes only. It does not import Data, Feature Core,
Qlib or a training backend. A verified store belongs to one reader/batch;
descriptors are deduplicated there, never by a persisted trusted flag.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import math
import sys
import json
from hashlib import sha256
import time
import weakref

from .stock_artifacts import digest, file_digest, _read, _verify_ref
from .stock_fold_inputs import require, ordered, file_fingerprint, validate_spec
from .stock_label_contracts import _instant, _session, _eligible_reason, NORMALIZATION_SPEC

DTYPES = {'float64_le': '<f8', 'bool_u8': 'u1', 'int64_le': '<i8',
          'uint64_le': '<u8', 'int32_le': '<i4', 'uint8': 'u1'}
BUFFER_FIELDS = {'path', 'file_digest', 'dtype', 'shape', 'buffer_digest'}
PARTITION_FIELDS = {'table', 'fold_spec_ref', 'row_index_ref', 'schema_digest',
    'row_offset', 'row_count', 'columns', 'buffers', 'metadata', 'partition_ref'}
FEATURE_INDEX_FIELDS = {'contract_version', 'definition', 'definition_ref', 'status',
    'qlib_view', 'qlib_manifest', 'schema', 'schema_digest', 'row_index',
    'source_selection', 'partitions', 'feature_inputs_ref', 'content_digest'}
FEATURE_BUFFERS = {'values', 'value_validity', 'available_at_utc_us', 'available_at_validity'}
_TOKEN = object()


def _ref(value):
    return type(value) is str and len(value) == 71 and value.startswith('sha256:') and all(
        c in '0123456789abcdef' for c in value[7:])


def _us(value):
    instant = _instant(value)
    delta = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _fields(value, fields, message):
    require(type(value) is dict and set(value) == set(fields), message)


class VerifiedMatrixStore:
    """One admission, one file hash/decode, and readonly mmap ownership."""
    def __init__(self, *, maximum_source_bytes=8*1024**3, maximum_matrix_bytes=512*1024**2,
                 maximum_parent_bytes=64*1024**2, _path_resolver=None):
        import numpy as np
        self.np = np
        for value in (maximum_source_bytes, maximum_matrix_bytes, maximum_parent_bytes):
            require(type(value) is int and value > 0, 'positive matrix byte budget required')
        self.maximum_source_bytes = maximum_source_bytes
        self.maximum_matrix_bytes = maximum_matrix_bytes
        self.maximum_parent_bytes = maximum_parent_bytes
        self._resolve = _path_resolver or (lambda p:Path(p))
        self.descriptors = {}; self.fingerprints = {}; self.json = {}; self.arrays = {}
        self._hashes={}; self._decoded={}
        self.content = {}; self.closed = False; self.borrowers = 0; self.resident_bytes = 0; self.lease_bytes=0
        self.metrics = {'unique_descriptor_admissions': 0, 'file_hash_calls': 0,
            'json_decode_calls': 0, 'mmap_opens': 0, 'source_bytes': 0,
            'hash_bytes': 0, 'common_key_index_builds': 0, 'fold_projection_calls': 0,
            'projection_bytes': 0, 'data_read_calls': 0, 'core_calls': 0,
            'train_calls': 0, 'predict_calls': 0, 'account_calls': 0}

    def _register(self, descriptor):
        require(not self.closed, 'matrix store is closed')
        path = descriptor.get('path')
        require(type(path) is str and Path(path).is_absolute() and not self._resolve(path).is_symlink(),
                'fixed absolute matrix file required')
        require(_ref(descriptor.get('file_digest')), 'matrix file digest required')
        if path in self.descriptors:
            old = self.descriptors[path]
            # Raw content-addressed files can appear as a 2-D partition and
            # the corresponding flat Core buffer. Each logical descriptor is
            # still checked below; only physical byte admission is shared.
            require(old == descriptor or (set(old) == BUFFER_FIELDS and
                    set(descriptor) == BUFFER_FIELDS and old['file_digest'] == descriptor['file_digest'] and
                    old['buffer_digest'] == descriptor['buffer_digest']), 'conflicting matrix descriptor')
            require(file_fingerprint(self._resolve(path)) == self.fingerprints[path], 'saved matrix source changed')
            return False
        mark = file_fingerprint(self._resolve(path))
        require(self._hash_once(path) == descriptor['file_digest'], 'matrix file digest mismatch')
        require(file_fingerprint(self._resolve(path)) == mark, 'matrix file changed during hash')
        self.descriptors[path] = deepcopy(descriptor); self.fingerprints[path] = mark
        return True

    def _hash_once(self,path):
        """Internal hook for the existing Qlib closure validator."""
        path=str(Path(path).absolute()); physical=self._resolve(path)
        require(not physical.is_symlink(),'fixed matrix file required')
        mark=file_fingerprint(physical)
        if path not in self._hashes:
            require(self.metrics['source_bytes']+mark[2]<=self.maximum_source_bytes,'matrix source byte budget exceeded')
            value=file_digest(physical)
            require(file_fingerprint(physical)==mark,'matrix file changed during hash')
            self._hashes[path]=value; self.fingerprints[path]=mark
            self.metrics['source_bytes']+=mark[2]; self.metrics['hash_bytes']+=mark[2]; self.metrics['file_hash_calls']+=1
            self.metrics['unique_descriptor_admissions']+=1
        require(file_fingerprint(physical)==self.fingerprints[path],'saved matrix source changed')
        return self._hashes[path]

    def _decode_once(self,path):
        path=str(Path(path).absolute()); physical=self._resolve(path)
        require(physical.stat().st_size<=self.maximum_parent_bytes,'matrix parent byte budget exceeded')
        if path not in self._decoded:
            self._decoded[path]=_read(physical); self.metrics['json_decode_calls']+=1
        return self._decoded[path]

    def read_json(self, descriptor, ref_key):
        _fields(descriptor, {'path', 'file_digest', ref_key}, 'exact matrix parent descriptor required')
        require(_ref(descriptor[ref_key]), 'matrix parent content ref required')
        fresh = self._register(descriptor); path = descriptor['path']
        if fresh:
            require(self.fingerprints[path][2] <= self.maximum_parent_bytes, 'matrix parent byte budget exceeded')
            value = self._decode_once(path)
            require(file_fingerprint(self._resolve(path)) == self.fingerprints[path], 'matrix parent changed during decode')
            _verify_ref(value, ref_key)
            require(value[ref_key] == descriptor[ref_key], 'matrix parent content mismatch')
            # Two exact canonical contents may share the decoded object.
            content_key = ref_key, descriptor[ref_key]
            if content_key in self.content:
                require(self.content[content_key] == value, 'conflicting canonical matrix content')
                value = self.content[content_key]
            else:
                self.content[content_key] = value
            self.json[path] = value
        require(path in self.json, 'released matrix proof cannot be readmitted inside a batch')
        return self.json[path]

    def drop_json(self, descriptor, ref_key):
        """Release a large proof after its compact admission has been built."""
        self.json.pop(descriptor['path'], None)
        self._decoded.pop(descriptor['path'],None)
        self.content.pop((ref_key, descriptor[ref_key]), None)

    def buffer(self, descriptor):
        _fields(descriptor, BUFFER_FIELDS, 'exact matrix buffer descriptor required')
        require(descriptor['dtype'] in DTYPES and type(descriptor['shape']) is list and
                bool(descriptor['shape']) and all(type(n) is int and n >= 0 for n in descriptor['shape']),
                'matrix buffer dtype/shape required')
        require(descriptor['buffer_digest'] == descriptor['file_digest'], 'raw buffer digest mismatch')
        self._register(descriptor); path = descriptor['path']
        key = path,descriptor['dtype'],tuple(descriptor['shape'])
        if key not in self.arrays:
            dtype = self.np.dtype(DTYPES[descriptor['dtype']])
            size = math.prod(descriptor['shape']) * dtype.itemsize
            require(size == self.fingerprints[path][2], 'matrix buffer byte length mismatch')
            if size:
                array = self.np.memmap(self._resolve(path), dtype=dtype, mode='r', shape=tuple(descriptor['shape']))
                self.metrics['mmap_opens'] += 1
            else:
                array = self.np.empty(descriptor['shape'], dtype=dtype); array.flags.writeable = False
            require(not array.flags.writeable, 'readonly matrix buffer required')
            if descriptor['dtype'] == 'bool_u8':
                require(bool(((array == 0) | (array == 1)).all()), 'matrix bool bytes must be 0/1')
            self.arrays[key] = array
        return self.arrays[key]

    def check(self):
        require(not self.closed, 'matrix store is closed')
        require(all(file_fingerprint(self._resolve(p)) == mark for p, mark in self.fingerprints.items()),
                'saved matrix source changed; initialize a fresh batch')

    def close(self):
        require(not self.closed,'matrix store is closed')
        require(self.borrowers == 0, 'matrix backing still borrowed')
        for array in self.arrays.values():
            if hasattr(array, '_mmap'): array._mmap.close()
        self.arrays.clear(); self.json.clear(); self.content.clear(); self._decoded.clear(); self.closed = True

    def __enter__(self): self.check(); return self
    def __exit__(self,*args): self.close()


def _row_index(value):
    _fields(value, {'contract_version','sessions','security_ids','order','row_count','row_index_ref'},
            'exact matrix row index required')
    require(value['contract_version'] == 'stock_matrix_row_index_v1' and value['order'] == 'session_security',
            'matrix row index order required')
    ordered(value['sessions'], 'matrix sessions'); [_session(d) for d in value['sessions']]
    ordered(value['security_ids'], 'matrix securities')
    require(type(value['row_count']) is int and value['row_count'] ==
            len(value['sessions'])*len(value['security_ids']), 'complete matrix row index required')
    _verify_ref(value, 'row_index_ref')
    return value


def _schema(schema, columns=None):
    require(type(schema) is list and bool(schema), 'nonempty matrix schema required')
    for column in schema:
        _fields(column, {'name','dtype','unit','stage','missing'}, 'exact neutral matrix column required')
        require(column['dtype'] == 'float64' and type(column['name']) is str and
                column['missing'] in ('preserve','reject'), 'float64 matrix schema required')
    names = [c['name'] for c in schema]
    require(len(set(names)) == len(names) and (columns is None or names == columns), 'matrix column order mismatch')
    return names


class _CompactRows:
    """Retain row facts as codes rather than a full Python cell panel."""
    def __init__(self, rows, width, np):
        self.member = np.asarray([r['member'] for r in rows], dtype='?')
        self.dictionary = []; indexes = {}
        def code(value):
            key = digest(value)
            if key not in indexes:
                indexes[key] = len(self.dictionary); self.dictionary.append(deepcopy(value))
            return indexes[key]
        self.knowledge = np.asarray([code(r['knowledge_cutoff']) for r in rows], dtype='<u4')
        self.sources = np.asarray([code(r['source_refs']) for r in rows], dtype='<u4')
        require(all(len(r['reasons'])==len(r['availability'])==width for r in rows),'complete matrix row metadata required')
        # Intern the complete row vector. The same route supports any column
        # count, retaining exact strings/lists without a dense N×F code panel.
        self.reasons = np.asarray([code(r['reasons']) for r in rows], dtype='<u4')
        self.availability = np.asarray([code(r['availability']) for r in rows], dtype='<u4')
        self.bytes = sum(a.nbytes for a in (self.member,self.knowledge,self.sources,self.reasons,self.availability))+_resident_size(self.dictionary)
        for a in (self.member,self.knowledge,self.sources,self.reasons,self.availability): a.flags.writeable = False

    def row(self, ordinal, *, security, session, values, validity):
        return {'security_id':security,'session':session,'member':bool(self.member[ordinal]),
            'knowledge_cutoff':deepcopy(self.dictionary[int(self.knowledge[ordinal])]),
            'source_refs':deepcopy(self.dictionary[int(self.sources[ordinal])]),
            'values':values,'validity':validity,
            'reasons':deepcopy(self.dictionary[int(self.reasons[ordinal])]),
            'availability':deepcopy(self.dictionary[int(self.availability[ordinal])])}


def _resident_size(value):
    seen=set()
    def size(item):
        if id(item) in seen: return 0
        seen.add(id(item)); total=sys.getsizeof(item)
        if type(item) is dict: total+=sum(size(k)+size(v) for k,v in item.items())
        elif type(item) in (list,tuple,set): total+=sum(size(v) for v in item)
        return total
    return size(value)


def _validate_partitions(partitions, row_index, schemas, store, *, expected_fold_refs=None, metadata=True):
    require(type(partitions) is list and bool(partitions), 'matrix partitions required')
    admitted = []; coverage = {}
    for p in partitions:
        _fields(p, PARTITION_FIELDS, 'exact matrix partition required'); _verify_ref(p,'partition_ref')
        require(p['table'] in schemas and p['row_index_ref'] == row_index['row_index_ref'] and
                p['schema_digest'] == digest(schemas[p['table']]), 'matrix partition index/schema mismatch')
        columns = _schema(schemas[p['table']]); selected = p['columns']
        require(type(selected) is list and bool(selected) and len(set(selected)) == len(selected) and
                all(c in columns for c in selected) and selected == [c for c in columns if c in selected],
                'ordered matrix partition columns required')
        start,count = p['row_offset'],p['row_count']
        require(type(start) is int and type(count) is int and start >= 0 and count > 0 and
                start+count <= row_index['row_count'], 'matrix partition row range mismatch')
        if p['table'] == 'features': require(p['fold_spec_ref'] is None, 'Feature partition cannot bind a fit')
        else: require(expected_fold_refs is None or p['fold_spec_ref'] in expected_fold_refs,
                      'matrix label partition outside fit definition')
        key = p['table'],p['fold_spec_ref']
        for column in selected:
            intervals = coverage.setdefault((key,column), [])
            require(all(start+count <= a or start >= b for a,b in intervals), 'duplicate matrix cell')
            intervals.append((start,start+count))
        _fields(p['buffers'], FEATURE_BUFFERS, 'complete value/mask/clock buffers required')
        arrays = {name:store.buffer(desc) for name,desc in p['buffers'].items()}
        shape = (count,len(selected))
        require(all(a.shape == shape for a in arrays.values()), 'matrix partition buffer shape mismatch')
        for name,dtype in (('values','float64_le'),('value_validity','bool_u8'),
                           ('available_at_utc_us','int64_le'),('available_at_validity','bool_u8')):
            require(p['buffers'][name]['dtype'] == dtype, 'matrix partition buffer dtype mismatch')
        valid = arrays['value_validity'].astype(bool, copy=False); values = arrays['values']
        require(bool(store.np.isfinite(values).all()) and bool((values[~valid] == 0).all()) and
                not bool(store.np.signbit(values[~valid]).any()), 'canonical finite/null matrix values required')
        clocks = arrays['available_at_utc_us']; clock_valid = arrays['available_at_validity'].astype(bool,copy=False)
        require(bool((clocks[~clock_valid] == 0).all()), 'canonical missing matrix clock required')
        proof = store.read_json(p['metadata'],'metadata_ref') if metadata else None
        admitted.append((p,arrays,proof))
    return admitted,coverage


def _complete_intervals(intervals, stop):
    previous = 0
    for start,end in sorted(intervals):
        require(start == previous, 'missing matrix cell'); previous = end
    require(previous == stop, 'incomplete matrix cell coverage')


class FeatureMatrixInputs:
    """A validated readonly Feature index with bounded row/column projection."""
    def __init__(self, token, *, path,index,store,row_index,blocks):
        require(token is _TOKEN, 'Feature matrix requires saved validation')
        self.path=Path(path); self.reused=False; self._index=index; self._store=store; self._row_index=row_index; self._blocks=blocks
        self._sessions={d:i for i,d in enumerate(row_index['sessions'])}
        self._securities={s:i for i,s in enumerate(row_index['security_ids'])}

    def to_dict(self): self._store.check(); return deepcopy(self._index)
    @property
    def identity(self): return self.to_dict()['feature_inputs_ref']
    @property
    def descriptor(self):
        path=str((self.path/'index.json').resolve())
        return {'path':path,'file_digest':self._store.descriptors[path]['file_digest'],'feature_inputs_ref':self.identity}
    @property
    def metrics(self): return deepcopy(self._store.metrics)
    def close(self): self._store.close(); self._blocks.clear()
    def __enter__(self): self._store.check(); return self
    def __exit__(self,*args): self.close()
    def offsets(self, keys):
        width=len(self._securities)
        require(all(s in self._securities and d in self._sessions for s,d in keys), 'Feature key outside matrix index')
        return self._store.np.asarray([self._sessions[d]*width+self._securities[s] for s,d in keys],dtype='<u8')
    def keys(self, offsets):
        ids=self._row_index['security_ids']; days=self._row_index['sessions']; width=len(ids)
        return [[ids[int(i)%width],days[int(i)//width]] for i in offsets]
    def project(self, offsets, columns=None):
        self._store.check(); np=self._store.np
        offsets=np.asarray(offsets,dtype='<u8'); allcolumns=self._index['definition']['spec']['ordered_features']
        columns=allcolumns if columns is None else columns
        require(type(columns) is list and bool(columns) and len(set(columns))==len(columns) and
                all(c in allcolumns for c in columns), 'Feature projection columns required')
        require(offsets.ndim==1 and bool((offsets < self._row_index['row_count']).all()),'Feature projection offsets invalid')
        required=len(offsets)*len(columns)*8
        require(required+len(offsets)*len(columns)+self._store.resident_bytes+self._store.lease_bytes<=self._store.maximum_matrix_bytes,
                'Feature projection matrix budget exceeded')
        values=np.empty((len(offsets),len(columns)),dtype='<f8'); validity=np.empty(values.shape,dtype='?')
        for block in self._blocks:
            start,count=block['start'],block['count']; indexes=np.nonzero((offsets>=start)&(offsets<start+count))[0]
            if not len(indexes): continue
            local=offsets[indexes].astype('int64')-start
            for p,arrays in block['parts']:
                for j,column in enumerate(p['columns']):
                    if column in columns:
                        k=columns.index(column); values[indexes,k]=arrays['values'][local,j]
                        validity[indexes,k]=arrays['value_validity'][local,j]
        self._store.metrics['projection_bytes']+=values.nbytes+validity.nbytes
        self._store.check(); return values,validity
    def row_metadata(self, offsets):
        values,validity=self.project(offsets); out=[]; ids=self._row_index['security_ids']; days=self._row_index['sessions']; width=len(ids)
        for ordinal,offset in enumerate(offsets):
            offset=int(offset)
            block=next(b for b in self._blocks if b['start']<=offset<b['start']+b['count'])
            vals=[float(v) if valid else None for v,valid in zip(values[ordinal],validity[ordinal])]
            out.append(block['rows'].row(offset-block['start'],security=ids[offset%width],session=days[offset//width],
                values=vals,validity=[bool(v) for v in validity[ordinal]]))
        return out


def admit_feature_matrix_parts(store, *, spec,view,schema,row_index,partitions,complete=False,universe_id=None):
    from .stock_feature_inputs import _validate_feature_matrix_block, _feature_contents
    admitted,coverage=_validate_partitions(partitions,row_index,{'features':schema},store,metadata=False)
    stop = row_index['row_count'] if complete else max((p['row_offset']+p['row_count'] for p in partitions), default=0)
    for column in spec['ordered_features']: _complete_intervals(coverage.get((('features',None),column),[]),stop)
    groups={}
    for p,arrays,_ in admitted:
        key=p['row_offset'],p['row_count'],digest(p['metadata'])
        groups.setdefault(key,[]).append((p,arrays))
    blocks=[]; reachable=set(); parentrefs={}; compact_bytes=0
    for (start,count,_),parts in sorted(groups.items()):
        metadata=store.read_json(parts[0][0]['metadata'],'metadata_ref'); rows=deepcopy(metadata['rows'])
        require(len(rows)==count and start%len(spec['universe'])==0 and count%len(spec['universe'])==0,'complete day Feature block required')
        for i,row in enumerate(rows):
            index=start+i; require((row['security_id'],row['session'])==(spec['universe'][index%len(spec['universe'])],spec['feature_sessions'][index//len(spec['universe'])]),'Feature metadata row order/key mismatch')
            row['values']=[None]*len(spec['ordered_features'])
        for p,arrays in parts:
            for j,column in enumerate(p['columns']):
                k=spec['ordered_features'].index(column)
                for i,row in enumerate(rows):
                    valid=bool(arrays['value_validity'][i,j]); row['values'][k]=float(arrays['values'][i,j]) if valid else None
                    require(row['validity'][k] is valid,'Feature binary validity mismatch')
                    at=row['availability'][k]; present=bool(arrays['available_at_validity'][i,j])
                    require(present is (at is not None) and (not present or int(arrays['available_at_utc_us'][i,j])==_us(at)), 'Feature binary clock mismatch')
        _validate_feature_matrix_block(metadata,rows,spec,view,universe_id=universe_id)
        compact=_CompactRows(rows,len(spec['ordered_features']),store.np); compact_bytes+=compact.bytes
        require(compact_bytes<=store.maximum_matrix_bytes,'Feature compact metadata budget exceeded')
        blocks.append({'start':start,'count':count,'parts':parts, 'rows':compact,
            'row_references':deepcopy(metadata['row_references']),
            '_source_selection_rows':deepcopy(_feature_contents(metadata['input_evidence'])[1]),
            '_original_feature_ref':metadata['original_feature_ref'],'_sessions':list(metadata['sessions'])})
        for ref,content in metadata['contents'].items():
            require(_ref(ref) and digest(content)==ref,'Feature source evidence content hash mismatch')
            reachable.add(ref)
        parentrefs.update(metadata['row_references'])
        store.drop_json(parts[0][0]['metadata'],'metadata_ref')
        del metadata,rows
    store.metrics['compact_metadata_bytes']=compact_bytes
    store.resident_bytes+=compact_bytes
    require(store.resident_bytes<=store.maximum_matrix_bytes,'Feature compact metadata budget exceeded')
    return row_index['sessions'][:stop//len(spec['universe'])],blocks,set(reachable),parentrefs


def load_feature_matrix_index(path, limits=None, *, store=None, _index=None, _index_fingerprint=None):
    """Strict v2 dispatch target. No numerical or provider operations occur."""
    from .stock_feature_inputs import _spec, _qlib, _qlib_scope, _validate_feature_matrix_block
    begin=time.perf_counter(); path=Path(path); index_path=path/'index.json'
    if limits is not None:
        _fields(limits,{'maximum_parent_bytes'},'positive parent byte limit required')
    own=store is None
    store=store or VerifiedMatrixStore(maximum_parent_bytes=(limits or {}).get('maximum_parent_bytes',64*1024**2))
    try:
        # The index file descriptor is explicit in a batch; a standalone caller
        # supplies its fixed path and the reader establishes the descriptor.
        index_name=str(index_path.resolve())
        if _index_fingerprint is not None:
            require(file_fingerprint(index_path)==_index_fingerprint,'Feature index changed during dispatch')
        existing=store.descriptors.get(index_name)
        if existing is not None:
            require(file_fingerprint(index_name)==store.fingerprints[index_name],'Feature index changed')
            value=store.json[index_name]
        else:
            value=_read(index_path) if _index is None else deepcopy(_index)
            canonical=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
            index_desc={'path':index_name,'file_digest':'sha256:'+sha256(canonical).hexdigest(),
                        'feature_inputs_ref':value.get('feature_inputs_ref')}
            # Feature ref is a defined projection, not digest(index minus ref).
            store._register(index_desc); store.json[index_name]=value; store.metrics['json_decode_calls']+=1
        _fields(value,FEATURE_INDEX_FIELDS,'exact Feature matrix index required')
        require(value['contract_version']=='stock_feature_inputs_v2' and value['status']=='COMPLETE', 'complete Feature matrix index required')
        _verify_ref(value,'content_digest'); definition=value['definition']
        _fields(definition,{'spec','storage_options','implementation_sources','implementation_ref','environment'},'exact matrix Feature definition required')
        require(definition['implementation_ref']==digest(definition['implementation_sources']) and
                value['definition_ref']==digest(definition),'matrix Feature implementation/definition mismatch')
        options=definition['storage_options']; _fields(options,{'layout','row_block_sessions','column_block','maximum_resident_bytes'},'exact matrix storage options required')
        require(options['layout']=='matrix_v1' and all(type(options[k]) is int and options[k]>0 for k in options if k!='layout'), 'matrix storage options invalid')
        spec=definition['spec']; universe_id=_spec(spec,reader=store.read_json)
        view=_qlib(value['qlib_manifest'],{'maximum_parent_bytes':store.maximum_parent_bytes},fingerprints=store.fingerprints,
                   _file_hasher=store._hash_once,_json_reader=store._decode_once)
        require(view==value['qlib_view'],'matrix Qlib reference mismatch'); _qlib_scope(view,spec,universe_id)
        require(value['feature_inputs_ref']==digest({k:value[k] for k in ('contract_version','definition_ref','qlib_manifest',
            'schema_digest','row_index','source_selection','partitions')}),'matrix Feature identity mismatch')
        require(value['schema_digest']==digest(value['schema']),'matrix Feature schema digest mismatch'); _schema(value['schema'],spec['ordered_features'])
        row_index=_row_index(store.read_json(value['row_index'],'row_index_ref'))
        require(row_index['sessions']==spec['feature_sessions'] and row_index['security_ids']==spec['universe'],'matrix Feature scope mismatch')
        selection=store.read_json(value['source_selection'],'source_selection_ref')
        _fields(selection,{'contract_version','definition_ref','feature_rows','source_selection_ref'},'exact Feature source selection required')
        require(selection['contract_version']=='stock_feature_source_selection_v1' and selection['definition_ref']==value['definition_ref'], 'Feature source-selection definition mismatch')
        covered,blocks,reachable,parentrefs=admit_feature_matrix_parts(store,spec=spec,view=view,schema=value['schema'],row_index=row_index,partitions=value['partitions'],complete=True,universe_id=universe_id)
        require(selection['feature_rows']==[r for b in blocks for r in b['_source_selection_rows']],
                'Feature source selection differs from actual proof')
        require([r['session'] for r in selection['feature_rows']]==spec['feature_sessions'],'Feature selection date coverage mismatch')
        for r in selection['feature_rows']:
            _fields(r,{'session','cutoff','history_sessions','adjustment_anchor','query_refs','selected_versions_ref'},'exact Feature selection row required')
            require(_instant(r['cutoff'])==_instant(spec['cutoff_by_session'][r['session']]) and r['adjustment_anchor']==r['session'] and
                    r['history_sessions'][-1]==r['session'] and all(q in reachable for q in r['query_refs']) and
                    r['selected_versions_ref'] in reachable,'Feature source selection closure mismatch')
        # Keep leaf hashes, not the large proof graphs after full admission.
        store.metrics['common_key_index_builds']+=1
        store.metrics['initialization_seconds']=time.perf_counter()-begin; store.check()
        result=FeatureMatrixInputs(_TOKEN,path=path,index=deepcopy(value),store=store,row_index=row_index,blocks=blocks)
        result._source_hashes=set(reachable); result._parents_by_session=parentrefs
        return result
    except BaseException:
        if own and not store.closed and not store.borrowers:
            for array in store.arrays.values():
                if hasattr(array,'_mmap'): array._mmap.close()
        raise


_CORE_OUT_BUFFERS={'values':'float64','value_validity':'bool','value_reason_codes':'int32',
                   'available_at_utc_us':'int64','source_codes':'int32'}
_CORE_IN_BUFFERS={'values':'float64','value_validity':'bool','value_reason_codes':'int32',
    'fact_available_at_utc_us':'int64','reference_member':'bool','reference_available_at_utc_us':'int64',
    'selection_cutoff_utc_us':'int64','fact_source_codes':'int32','reference_source_codes':'int32'}
_CORE_DTYPE={'float64':'float64_le','bool':'bool_u8','int32':'int32_le','int64':'int64_le'}


def _core_buffers(document, descriptors, types, store, count, days):
    _fields(descriptors,types,'complete saved Core physical buffers required')
    arrays={}
    for name,dtype in types.items():
        desc=descriptors[name]; array=store.buffer(desc)
        expected=days if name=='selection_cutoff_utc_us' else count
        require(desc['dtype']==_CORE_DTYPE[dtype] and math.prod(desc['shape'])==expected,
                'Core physical buffer dtype/shape mismatch')
        canonical={'dtype':dtype,'shape':[expected],'bytes_digest':desc['buffer_digest']}
        require(document[name]==canonical,'Core canonical/physical buffer mismatch')
        arrays[name]=array.reshape(-1)
    return arrays


def _source_map(sources, days):
    require(type(sources) is dict and set(sources)==set(days),'Core source date coverage mismatch')
    for day in days:
        group=sources[day]; _fields(group,{'bindings','source_sets'},'exact Core source group required')
        bindings=group['bindings']; sets=group['source_sets']
        require(type(bindings) is list and bool(bindings),'Core source bindings required')
        ids=[]
        for binding in bindings:
            _fields(binding,{'id','data_ref','view_ref','revision_policy','qualification','availability_basis'},'exact Core source binding required')
            require(_ref(binding['data_ref']) and _ref(binding['view_ref']) and
                    binding['qualification'] in ('verified','observed','best_effort','synthetic'),'Core source binding ref mismatch')
            ids.append(binding['id'])
        require(ids==sorted(set(ids)) and all(type(v) is str and v for v in ids),'ordered Core sources required')
        require(type(sets) is list and bool(sets) and all(type(s) is list and bool(s) and
                s==sorted(set(s)) and set(s)<=set(ids) for s in sets) and
                [tuple(s) for s in sets]==sorted(set(tuple(s) for s in sets)),'Core source set mismatch')


def validate_saved_core_result(wrapper, store, *, core_input):
    """Verify stored output and exact input/dependency closure; never compute CS."""
    _fields(wrapper,{'contract_version','result','buffers','core_result_artifact_ref'},'exact Core result wrapper required')
    require(wrapper['contract_version']=='stock_matrix_core_result_v1','unsupported Core result wrapper')
    _verify_ref(wrapper,'core_result_artifact_ref'); result=wrapper['result']; meta=result['metadata']
    _fields(result,{'contract_version','sessions','security_ids','schema','values','value_validity','value_reason_codes',
        'reason_dictionary','available_at_utc_us','source_bindings_by_session','source_codes','metadata'},'exact canonical Core output required')
    require(result['contract_version']=='core_cs_zscore_batch_result_v1','unsupported canonical Core output')
    days=ordered(result['sessions'],'Core result sessions'); securities=ordered(result['security_ids'],'Core result securities')
    count=len(days)*len(securities); _schema(result['schema'])
    outputs=_core_buffers(result,wrapper['buffers'],_CORE_OUT_BUFFERS,store,count,len(days))
    _fields(meta,{'input_ref','numeric_input_ref','spec_ref','keys_ref','schema_ref','source_ref','context_ref',
                 'values_digest','flags_digest','clocks_digest','implementation_ref','result_ref'},'exact Core output metadata required')
    require(all(_ref(v) for v in meta.values()),'Core output references required')
    canon=deepcopy(result); canon['metadata'].pop('result_ref')
    require(meta['result_ref']==digest(canon) and meta['values_digest']==digest(result['values']) and
            meta['flags_digest']==digest({k:result[k] for k in ('reason_dictionary','value_validity','value_reason_codes')}) and
            meta['clocks_digest']==digest(result['available_at_utc_us']),'saved Core result digest mismatch')
    _fields(core_input,{'input','buffers'},'complete saved Core input closure required'); inp=core_input['input']
    _fields(inp,set(_CORE_IN_BUFFERS)|{'contract_version','calendar_ref','schema','output_schema','sessions','security_ids',
                                   'reason_dictionary','source_bindings_by_session'},'exact canonical Core input required')
    require(inp['contract_version']=='core_cs_zscore_batch_input_v1' and inp['sessions']==days and
            inp['security_ids']==securities and inp['output_schema']==result['schema'],'Core input/output grid/schema mismatch')
    _schema(inp['schema']); _schema(inp['output_schema'])
    require(len(inp['schema'])==len(inp['output_schema'])==1 and inp['schema'][0]['stage']=='fact' and
            inp['output_schema'][0]['stage']=='cross_sectional' and inp['output_schema'][0]['unit']=='dimensionless' and
            inp['output_schema'][0]['missing']=='preserve','Core input/output column admission mismatch')
    inputs=_core_buffers(inp,core_input['buffers'],_CORE_IN_BUFFERS,store,count,len(days))
    reasons=inp['reason_dictionary']; require(type(reasons) is list and reasons and reasons[0] is None and
        reasons[1:]==sorted(set(reasons[1:])) and all(type(v) is str and v for v in reasons[1:]),'Core input reasons invalid')
    keys_ref=digest({'sessions':days,'security_ids':securities,'order':'session_security'})
    spec_ref=digest({'abi':'axiom.feature/1','semantics':'axiom.operators/1','operator':'cs_zscore','operator_version':'1',
                     'params':NORMALIZATION_SPEC['params'],'clock_projection':'ceil_to_core_second_v1'})
    schema_ref=digest({'schema':inp['schema'],'output_schema':inp['output_schema']})
    numeric={'keys_ref':keys_ref,'schema_ref':schema_ref,'spec_ref':spec_ref,'reason_dictionary':reasons,
             **{k:inp[k] for k in ('values','value_validity','value_reason_codes','reference_member')}}
    source={'source_bindings_by_session':inp['source_bindings_by_session'],**{k:inp[k] for k in ('fact_source_codes','reference_source_codes')}}
    context={'calendar_ref':inp['calendar_ref'],'keys_ref':keys_ref,**{k:inp[k] for k in
        ('reference_member','reference_available_at_utc_us','selection_cutoff_utc_us')}}
    require(meta['input_ref']==digest(inp) and meta['numeric_input_ref']==digest(numeric) and
            meta['keys_ref']==keys_ref and meta['spec_ref']==spec_ref and meta['schema_ref']==schema_ref and
            meta['source_ref']==digest(source) and meta['context_ref']==digest(context),'Core saved input identity mismatch')
    _source_map(inp['source_bindings_by_session'],days); _source_map(result['source_bindings_by_session'],days)
    require(result['reason_dictionary']==[None,'REFERENCE_MISSING'],'Core output reasons mismatch')
    for i,day in enumerate(days):
        sources=inp['source_bindings_by_session'][day]; out_sources=result['source_bindings_by_session'][day]
        require(sources['bindings']==out_sources['bindings'],'Core output bindings mismatch')
        cutoff=int(inputs['selection_cutoff_utc_us'][i]); start=i*len(securities); stop=start+len(securities)
        dependency_sources=set(); dependency_clocks=[]
        for j in range(start,stop):
            valid=bool(inputs['value_validity'][j]); reason=int(inputs['value_reason_codes'][j])
            require(valid or inp['schema'][0]['missing']=='preserve','Core input schema rejects missing values')
            require(0<=reason<len(reasons) and (reason==0)==valid and math.isfinite(float(inputs['values'][j])) and
                    (valid or (float(inputs['values'][j])==0 and not store.np.signbit(inputs['values'][j]))), 'Core input value/reason invalid')
            fc=int(inputs['fact_source_codes'][j]); rc=int(inputs['reference_source_codes'][j])
            require(0<=fc<len(sources['source_sets']) and 0<=rc<len(sources['source_sets']) and
                    len(sources['source_sets'][rc])==1,'Core input source code invalid')
            fact=int(inputs['fact_available_at_utc_us'][j]); reference=int(inputs['reference_available_at_utc_us'][j])
            require(fact<=cutoff and reference<=cutoff,'Core native clock exceeds cutoff')
            dependency_clocks.append(-((-reference)//1000000)*1000000)
            dependency_sources.update(sources['source_sets'][rc])
            if inputs['reference_member'][j]:
                dependency_clocks.append(-((-fact)//1000000)*1000000)
                dependency_sources.update(sources['source_sets'][fc])
        for j in range(start,stop):
            valid=bool(outputs['value_validity'][j]); reason=int(outputs['value_reason_codes'][j])
            require(reason==(0 if valid else 1) and math.isfinite(float(outputs['values'][j])) and
                    (valid or float(outputs['values'][j])==0 and not store.np.signbit(outputs['values'][j])), 'Core output value/reason invalid')
            require(not valid or inputs['reference_member'][j] and inputs['value_validity'][j],'Core invalid/excluded output cannot become valid')
            fact=int(inputs['fact_available_at_utc_us'][j]); expected=max(dependency_clocks+[-((-fact)//1000000)*1000000])
            require(int(outputs['available_at_utc_us'][j])==expected and expected<=-((-cutoff)//1000000)*1000000,
                    'Core output dependency clock mismatch')
            code=int(outputs['source_codes'][j]); expected_sources=sorted(dependency_sources|set(sources['source_sets'][int(inputs['fact_source_codes'][j])]))
            require(0<=code<len(out_sources['source_sets']) and out_sources['source_sets'][code]==expected_sources,
                    'Core output dependency sources mismatch')
    return inputs,outputs


def _selector(desc, *, role,inputs,spec,view,row_index,store):
    value=store.read_json(desc,'selector_ref')
    _fields(value,{'contract_version','prepared_view_ref','row_index_ref','schema_digest','role','fold_spec_ref',
                  'encoding','row_count','payload','keys_digest','selector_ref'},'exact matrix selector required')
    require(value['contract_version']=='stock_matrix_selector_v1' and value['role']==role and
            value['prepared_view_ref']==view['prepared_view_ref'] and value['row_index_ref']==row_index['row_index_ref'] and
            value['schema_digest']==view['schema_digest'] and value['fold_spec_ref']==digest(spec),'matrix selector binding mismatch')
    np=store.np; payload=store.buffer(value['payload']); count=row_index['row_count']
    if value['encoding']=='offsets_u64_le':
        require(value['payload']['dtype']=='uint64_le' and payload.ndim==1,'selector offset dtype/shape mismatch')
        offsets=payload
        require(bool((offsets<count).all()) and (len(offsets)<2 or bool((offsets[1:]>offsets[:-1]).all())),
                'unique ordered selector offsets required')
    else:
        require(value['encoding']=='bitmap_lsb0' and value['payload']['dtype']=='uint8' and
                payload.shape==((count+7)//8,), 'selector bitmap dtype/shape mismatch')
        if count%8: require(int(payload[-1])>>(count%8)==0,'selector bitmap padding must be zero')
        offsets=np.flatnonzero(np.unpackbits(payload,bitorder='little')[:count]).astype('<u8')
        offsets.flags.writeable=False
    require(type(value['row_count']) is int and value['row_count']==len(offsets),'selector selected count mismatch')
    width=len(row_index['security_ids'])
    keys=[[row_index['security_ids'][int(i)%width],row_index['sessions'][int(i)//width]] for i in offsets]
    require(value['keys_digest']==digest(keys),'selector restored keys mismatch')
    return offsets,value


_TARGET_FIELDS={'security_id','feature_session','start_session','end_session','return',
                'label_available_at','valid','invalid_reason','source_refs'}
_NORMALIZED_FIELDS={'normalized_return','normalized_valid','normalized_available_at',
                    'normalization_reason','normalization_source_refs'}


class _TargetBlock:
    def __init__(self, rows, *, start,table,np):
        self.start=start; self.count=len(rows); self.table=table
        self.dictionary=[]; lookup={}
        def code(value):
            key=digest(value)
            if key not in lookup: lookup[key]=len(self.dictionary); self.dictionary.append(deepcopy(value))
            return lookup[key]
        self.codes={name:np.asarray([code(r[name]) for r in rows],dtype='<u4') for name in
                    ('start_session','end_session','label_available_at','invalid_reason','source_refs')}
        self.raw=np.asarray([float(r['return']) if r['return'] is not None else 0.0 for r in rows],dtype='<f8')
        self.raw_present=np.asarray([r['return'] is not None for r in rows],dtype='?')
        self.valid=np.asarray([r['valid'] for r in rows],dtype='?')
        self.normalized=None
        if table=='training_normalized_labels':
            self.normalized={name:np.asarray([code(r[name]) for r in rows],dtype='<u4') for name in
                ('normalized_available_at','normalization_reason','normalization_source_refs')}
            self.norm=np.asarray([float(r['normalized_return']) if r['normalized_return'] is not None else 0.0 for r in rows],dtype='<f8')
            self.norm_valid=np.asarray([r['normalized_valid'] for r in rows],dtype='?')
        for array in [*self.codes.values(),self.raw,self.raw_present,self.valid,
                      *([] if self.normalized is None else [*self.normalized.values(),self.norm,self.norm_valid])]:
            array.flags.writeable=False
        self.bytes=sum(a.nbytes for a in [*self.codes.values(),self.raw,self.raw_present,self.valid,
            *([] if self.normalized is None else [*self.normalized.values(),self.norm,self.norm_valid])])+_resident_size(self.dictionary)

    def row(self, offset,key):
        i=int(offset)-self.start
        require(0<=i<self.count,'Target key outside matrix block')
        result={'security_id':key[0],'feature_session':key[1],'return':float(self.raw[i]) if self.raw_present[i] else None,
                'valid':bool(self.valid[i]),**{name:deepcopy(self.dictionary[int(array[i])]) for name,array in self.codes.items()}}
        if self.normalized is not None:
            result.update(normalized_return=float(self.norm[i]) if self.norm_valid[i] else None,
                          normalized_valid=bool(self.norm_valid[i]),
                          **{name:deepcopy(self.dictionary[int(array[i])]) for name,array in self.normalized.items()})
        return result


def _find_core_input(contents, input_ref):
    found=[]
    for value in contents.values():
        if type(value) is dict and set(value)=={'input','buffers'} and digest(value['input'])==input_ref:
            found.append(value)
    require(len(found)==1,'saved Core input closure missing or ambiguous')
    return found[0]


def _outcome_query(context,calendar):
    """Read-only counterpart of the existing Label query admission."""
    query=context['query']; derived=context.get('derivation')
    require(context.get('domain')=='market_daily' and type(query['fields']) is list and len(query['fields'])==2 and
        set(query['fields'])=={'open','close'} and
        query['price_basis']=='common_anchor_adjusted_v1' and query['purpose']=='label_outcomes',
        'original outcome-only adjusted query required')
    sessions=ordered(query['sessions'],'label Query sessions'); require(set(sessions)<=set(calendar),'label Query outside calendar')
    clocks=query['cutoff_by_session']; require(set(clocks)==set(sessions),'complete label Query cutoffs required')
    instants=set(map(_instant,clocks.values())); require(len(instants)==1,'common label outcome cutoff required')
    cutoff=next(iter(instants)); anchor=query['adjustment_anchor']
    require(anchor in calendar and type(derived) is dict and derived.get('recipe_version')=='common_anchor_price_v1' and
        derived.get('formula')=='price_t * factor_t / factor_anchor' and derived.get('anchor_session')==anchor and
        derived.get('decision_session')==sessions[-1] and _instant(derived.get('decision_cutoff'))==cutoff,
        'original label native anchor/derivation required')
    for name in ('price_query','factor_query'):
        native=derived.get(name); expected=set(sessions) if name=='price_query' else set(sessions)|{anchor}
        require(type(native) is dict and native.get('purpose')=='label_outcomes' and native.get('price_basis')=='unadjusted' and
            native.get('adjustment_anchor') is None and native.get('pit_policy')==query['pit_policy'] and
            set(native.get('symbols',[]))==set(query['symbols']) and set(native.get('sessions',[]))==expected and
            set(native.get('cutoff_by_session',{}))==expected and
            all(_instant(v)==cutoff for v in native['cutoff_by_session'].values()),'original native outcome Query mismatch')
        require(all((native.get('policy_by_session') or {}).get(d)==(query.get('policy_by_session') or {}).get(d) for d in sessions),
                'native outcome per-session policy mismatch')
        require({'open','close'}<=set(native['fields']) if name=='price_query' else derived['factor_field'] in native['fields'],
                'native outcome fields mismatch')
    return query,query['symbols'],anchor,cutoff


def _label_admission(partitions, *, view,feature,store,row_index,core_wrappers,common,fold_specs):
    """Admit one bounded Raw/normalized chunk at a time; retain compact targets."""
    from .stock_fold_inputs import validate_raw
    schemas={k:v for k,v in view['schema'].items() if k!='features'}
    if not partitions: return {},set(),{},[]
    admitted,coverage=_validate_partitions(partitions,row_index,schemas,store,
                                          expected_fold_refs=set(fold_specs),metadata=False)
    groups={}
    for p,arrays,_ in admitted:
        key=p['fold_spec_ref'],p['row_offset'],p['row_count']
        groups.setdefault(key,[]).append((p,arrays))
    targets={}; reachable=set(); raw_refs={}; width=len(row_index['security_ids']); np=store.np; core_cache={}; selection_rows=[]
    for (fold_ref,start,count),pieces in sorted(groups.items()):
        spec=fold_specs[fold_ref]; core_seen=set(); raw_seen=set()
        group_features=(feature.row_metadata(np.arange(start,start+count,dtype='<u8'))
                        if any(p['table']!='evaluation_raw_labels' for p,_ in pieces) else None)
        for p,arrays in pieces:
            metadata=store.read_json(p['metadata'],'metadata_ref')
            _fields(metadata,{'contract_version','role','fold_spec_ref','cutoff','rows','raw_build','core_result_refs','contents','metadata_ref'},
                    'exact matrix Target metadata required')
            role='evaluation' if p['table']=='evaluation_raw_labels' else 'training'
            cutoff=spec['evaluation_cutoff'] if role=='evaluation' else spec['fit_cutoff']
            require(metadata['contract_version']=='stock_matrix_label_metadata_v1' and metadata['role']==role and
                    metadata['fold_spec_ref']==fold_ref and _instant(metadata['cutoff'])==_instant(cutoff),
                    'Target metadata fit/role/cutoff mismatch')
            rows=metadata['rows']; require(type(rows) is list and len(rows)==count,'complete matrix Target rows required')
            raw=store.read_json(metadata['raw_build'],'label_ref')
            raw_index=validate_raw(raw,common,cutoff)
            query,_,anchor,native_cutoff=_outcome_query(raw['source_evidence']['context'],common['calendar'])
            require(native_cutoff==_instant(cutoff),'Target original Query current cutoff mismatch')
            if role=='training': raw_refs.setdefault(fold_ref,set()).add(raw['label_ref'])
            raw_seen.add((metadata['raw_build']['path'],raw['label_ref']))
            refs=metadata['core_result_refs']; require(type(refs) is list and len(set(refs))==len(refs),'ordered Target Core result refs required')
            if p['table']=='training_normalized_labels':
                require(bool(refs),'normalized Target requires saved Core result')
            else: require(not refs,'raw Target cannot claim normalization output')
            for ref,content in metadata['contents'].items():
                require(_ref(ref) and digest(content)==ref,'Target source evidence content hash mismatch'); reachable.add(ref)
            selected=[(ref,content) for ref,content in metadata['contents'].items() if type(content) is dict and
                      set(content)=={'context','records','field_meta'} and content['context']==raw['source_evidence']['context']]
            require(len(selected)==1,'Target actual selected-version evidence missing or ambiguous')
            selected_ref,selected_wire=selected[0]
            require(digest(selected_wire)==raw['source_ref'] and digest(selected_wire['records'])==raw['source_evidence']['records_ref'] and
                    digest(selected_wire['field_meta'])==raw['source_evidence']['field_meta_ref'], 'Target source evidence/raw original refs mismatch')
            qref=digest(query); require(metadata['contents'].get(qref)==query,'Target original logical Query not reachable')
            cohorts=[(ref,c) for ref,c in metadata['contents'].items() if type(c) is dict and
                     c.get('contract_version')=='stock_matrix_label_cohort_v1']
            require(len(cohorts)==(1 if role=='training' else 0),'Target cohort evidence mismatch')
            if cohorts:
                expected_reasons=[_eligible_reason(r,f,len(common['ordered_features']),_instant(cutoff)) for r,f in zip(rows,group_features)]
                expected_cohort={'contract_version':'stock_matrix_label_cohort_v1',
                    'feature_inputs_ref':view['definition']['feature_inputs']['feature_inputs_ref'],'raw_label_ref':raw['label_ref'],
                    'fold_spec_ref':fold_ref,'cutoff':metadata['cutoff'],'sessions':sorted({r['feature_session'] for r in rows}),
                    'keys':[[r['security_id'],r['feature_session']] for r in rows], 'eligibility_reasons':expected_reasons,
                    'eligible_keys':[[r['security_id'],r['feature_session']] for r,reason in zip(rows,expected_reasons) if reason is None]}
                require(cohorts[0][1]==expected_cohort,'Target current fit eligibility cohort mismatch')
            if not refs:
                selection_rows.append({'role':role,'fold_spec_ref':fold_ref,'cutoff':metadata['cutoff'],
                    'sessions':sorted({r['feature_session'] for r in rows}),'adjustment_anchor':anchor,
                    'query_refs':[qref],'selected_versions_ref':selected_ref,'cohort_ref':cohorts[0][0] if cohorts else None})
            feature_rows=None; core_cells={}
            if refs:
                feature_rows=group_features
                for ref in refs:
                    require(ref in core_wrappers,'Target Core result outside prepared view')
                    wrapper=core_wrappers[ref]; inp=_find_core_input(metadata['contents'],wrapper['result']['metadata']['input_ref'])
                    if ref not in core_cache:
                        core_cache[ref]=validate_saved_core_result(wrapper,store,core_input=inp)
                    in_arrays,out_arrays=core_cache[ref]
                    canon=inp['input']; require(canon['security_ids']==row_index['security_ids'] and
                        canon['calendar_ref']==digest({'contract_version':'stock_label_calendar_v1','sessions':common['calendar']}),
                        'normalized Core calendar/universe mismatch')
                    for i,day in enumerate(canon['sessions']):
                        require(day in row_index['sessions'],'Core normalized day outside row index')
                        day_start=row_index['sessions'].index(day)*width
                        if not start<=day_start<start+count: continue
                        bindings={b['id']:b for b in canon['source_bindings_by_session'][day]['bindings']}
                        require('raw_labels' in bindings and 'offline_eligibility' in bindings and
                            bindings['raw_labels']['data_ref']==raw['label_ref'] and bindings['raw_labels']['view_ref']==raw['label_ref'] and
                            bindings['offline_eligibility']['data_ref']==view['definition']['feature_inputs']['feature_inputs_ref'],
                            'normalized Core original source binding mismatch')
                        require(bindings['offline_eligibility']['view_ref']==cohorts[0][0],
                                'normalized Core eligibility source differs from current cohort')
                        require(int(in_arrays['selection_cutoff_utc_us'][i])==_us(cutoff),'Core normalization current fit cutoff mismatch')
                        for j,security in enumerate(canon['security_ids']):
                            off=row_index['sessions'].index(day)*width+j
                            require(start<=off<start+count and off not in core_cells,'overlapping/outside saved Core output')
                            k=i*width+j; original=raw_index[security,day]; frow=feature_rows[off-start]
                            reason=_eligible_reason(original,frow,len(common['ordered_features']),_instant(cutoff))
                            eligible=reason is None
                            require(bool(in_arrays['reference_member'][k]) is eligible and bool(in_arrays['value_validity'][k]) is eligible and
                                    int(in_arrays['reference_available_at_utc_us'][k])==_us(cutoff) and
                                    int(in_arrays['fact_available_at_utc_us'][k])==(_us(original['label_available_at']) if eligible else _us(cutoff)) and
                                    float(in_arrays['values'][k])==(float(original['return']) if eligible else 0.0),
                                    'normalized Core raw/Feature eligibility/clock mismatch')
                            require(canon['reason_dictionary'][int(in_arrays['value_reason_codes'][k])]==reason,'normalized Core original input reason mismatch')
                            info=canon['source_bindings_by_session'][day]
                            require(info['source_sets'][int(in_arrays['fact_source_codes'][k])]==(['raw_labels'] if eligible else ['offline_eligibility']) and
                                    info['source_sets'][int(in_arrays['reference_source_codes'][k])]==['offline_eligibility'],
                                    'normalized Core eligibility source mapping mismatch')
                            core_cells[off]=(reason,bool(out_arrays['value_validity'][k]),float(out_arrays['values'][k]),
                                             int(out_arrays['available_at_utc_us'][k]),ref)
                    core_seen.add(ref)
            for i,row in enumerate(rows):
                _fields(row,_TARGET_FIELDS|(_NORMALIZED_FIELDS if refs else set()),'exact thin matrix Target row required')
                offset=start+i; key=(row_index['security_ids'][offset%width],row_index['sessions'][offset//width])
                require((row['security_id'],row['feature_session'])==key and key in raw_index and
                        {k:row[k] for k in _TARGET_FIELDS}=={k:raw_index[key][k] for k in _TARGET_FIELDS},
                        'Target row key/raw parent mismatch')
                if refs:
                    require(offset in core_cells,'missing complete normalized Core output grid')
                    reason,valid,value,clock,ref=core_cells[offset]
                    normalized_valid=reason is None and valid
                    require(row['normalized_valid'] is normalized_valid and
                            row['normalized_return']==(value if normalized_valid else None) and
                            row['normalization_reason']==(None if normalized_valid else reason or 'NORMALIZATION_UNDEFINED') and
                            (row['normalized_available_at'] is not None)==normalized_valid and
                            (not normalized_valid or _us(row['normalized_available_at'])==clock), 'normalized Target/Core output mismatch')
                    value=row['normalized_return']; valid=row['normalized_valid']; at=row['normalized_available_at']
                    require(ref in row['normalization_source_refs'] and raw['label_ref'] in row['normalization_source_refs'],
                            'normalized Target output source linkage mismatch')
                else:
                    value=row['return']; valid=row['valid']; at=row['label_available_at']
                require(arrays['values'].shape[1]==1 and bool(arrays['value_validity'][i,0]) is valid and
                        float(arrays['values'][i,0])==(float(value) if valid else 0.0) and
                        bool(arrays['available_at_validity'][i,0]) is (at is not None) and
                        (at is None or int(arrays['available_at_utc_us'][i,0])==_us(at)), 'Target binary values/flags/clocks mismatch')
            target=_TargetBlock(rows,start=start,table=p['table'],np=np)
            target.core_refs=list(refs)
            store.resident_bytes+=target.bytes
            require(store.resident_bytes<=store.maximum_matrix_bytes,'Target compact metadata budget exceeded')
            targets.setdefault((fold_ref,p['table']),[]).append(target)
            store.drop_json(p['metadata'],'metadata_ref')
            del metadata,rows,raw_index,feature_rows,core_cells
        for path,ref in raw_seen:
            if path in store.json: store.drop_json({'path':path,'label_ref':ref},'label_ref')
        del group_features
    return targets,reachable,raw_refs,selection_rows


def _target_row(targets,table,fold_ref,offset,key):
    blocks=targets.get((fold_ref,table),[])
    matches=[b for b in blocks if b.start<=int(offset)<b.start+b.count]
    require(len(matches)==1,'missing/duplicate matrix Target key')
    return matches[0].row(offset,key)


class MatrixFoldProjection:
    """One active fold lease; closing the owning batch while borrowed fails."""
    def __init__(self,store,**values):
        self._store=store; self._closed=False; self.__dict__.update(values); store.borrowers+=1
        self._finalizer=weakref.finalize(self,MatrixFoldProjection._release,weakref.ref(store))
    @staticmethod
    def _release(reference):
        store=reference()
        if store is not None: store.borrowers-=1
    def close(self):
        if not self._closed:
            self._closed=True; self._finalizer()
            self.X=self.y=self.P=None
            self._store.lease_bytes-=getattr(self,'_lease_bytes',0)
            if getattr(self,'_owned_state',None) is not None: self._owned_state.close()
    def __enter__(self): require(not self._closed,'matrix fold projection is closed'); return self
    def __exit__(self,*args): self.close()


def _saved_inputs(inputs,spec,view):
    _fields(inputs,{'contract_version','prepared_view','fold_spec_ref','selectors','core_result_refs','input_ref'},
            'exact saved matrix inputs required')
    require(inputs['contract_version']=='stock_ml_saved_inputs_v2' and inputs['fold_spec_ref']==digest(spec),
            'saved matrix inputs/fold mismatch'); _verify_ref(inputs,'input_ref')
    _fields(inputs['selectors'],{'training','validation','inference','training_labels','evaluation_labels'},
            'exact fold selector set required')
    require(inputs['selectors']['validation'] is None,'fixed profile has no validation split')
    require(inputs['prepared_view']['prepared_view_ref']==view['prepared_view_ref'],'saved matrix prepared view mismatch')
    require(type(inputs['core_result_refs']) is list and bool(inputs['core_result_refs']) and
            len(set(inputs['core_result_refs']))==len(inputs['core_result_refs']) and
            all(_ref(v) for v in inputs['core_result_refs']),'explicit ordered fold Core refs required')


def _stream_ref(rows):
    h=sha256(); h.update(b'['); first=True
    for row in rows:
        if not first: h.update(b',')
        first=False; h.update(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
    h.update(b']'); return 'sha256:'+h.hexdigest()


class MatrixBatchState:
    def __init__(self,token,*,store,view,feature,row_index,targets,raw_refs,core_wrappers,selectors):
        require(token is _TOKEN,'matrix batch requires saved validation')
        self.store=store; self.view=view; self.feature=feature; self.row_index=row_index
        self.targets=targets; self.raw_refs=raw_refs; self.core_wrappers=core_wrappers; self.selectors=selectors

    def close(self):
        self.store.close(); self.targets.clear(); self.selectors.clear(); self.feature._blocks.clear()

    def project(self,inputs,spec):
        self.store.check(); view=self.view; common=deepcopy(view['definition']); fold_ref=digest(spec)
        key=digest(inputs),fold_ref
        require(key in self.selectors,'fold outside saved matrix batch definition')
        selected=self.selectors[key]; training_dates,prediction_dates=validate_spec(spec,common['calendar'])
        train=selected['training']; infer=selected['inference']; keys=self.feature.keys(train)
        columns=common['ordered_features']; np=self.store.np
        required=(len(train)+len(infer))*len(columns)*9+len(train)*8
        require(required+self.store.resident_bytes+self.store.lease_bytes<=self.store.maximum_matrix_bytes,
                'active fold matrix byte budget exceeded')
        X,flags=self.feature.project(train); require(bool(flags.all()) and bool(np.isfinite(X).all()),'selected training matrix invalid')
        y=np.asarray([_target_row(self.targets,'training_normalized_labels',fold_ref,off,key)['normalized_return']
                      for off,key in zip(train,keys)],dtype='<f8')
        require(bool(np.isfinite(y).all()),'selected training target nonfinite')
        rows=self.feature.row_metadata(infer); candidate_positions=[]; candidate_keys=[]
        for i,row in enumerate(rows):
            valid=(row['member'] and all(row['validity']) and all(type(v) in (int,float) and math.isfinite(v) for v in row['values']) and
                   any(a is not None for a in row['availability']))
            if valid:
                candidate_positions.append(i); candidate_keys.append([row['security_id'],row['session']])
        candidate_offsets=infer[np.asarray(candidate_positions,dtype='int64')]
        P,pflags=self.feature.project(candidate_offsets); require(bool(pflags.all()),'prediction candidate matrix invalid')
        excluded={}; width=len(common['universe']); dates=self.row_index['sessions']; calendar=common['calendar']
        selected_set=set(int(v) for v in train); expected=[]
        for day in training_dates:
            pos=calendar.index(day); immature=pos+5>=len(calendar) or calendar[pos+5]>_instant(spec['fit_cutoff']).date().isoformat()
            base=dates.index(day)*width
            for j,security in enumerate(common['universe']):
                offset=base+j; row=_target_row(self.targets,'training_normalized_labels',fold_ref,offset,[security,day])
                reason='LABEL_NOT_MATURE' if immature else (row['normalization_reason'] or 'NORMALIZATION_UNDEFINED') if not row['normalized_valid'] else None
                if reason is None: expected.append(offset)
                else: excluded[reason]=excluded.get(reason,0)+1
                require((offset in selected_set)==(reason is None),'training selector does not match mature eligible target grid')
        require(expected==[int(v) for v in train],'training selector key order mismatch')
        parents={d:deepcopy(self.feature._parents_by_session[d]) for d in sorted(set(training_dates+prediction_dates))}
        features={'contract_version':'stock_feature_slice_v3','input_manifest_ref':digest(inputs),
            'prepared_view_ref':view['prepared_view_ref'],'selectors':deepcopy(inputs['selectors']),
            'universe':common['universe'],'ordered_features':columns,'catalog_ref':common['catalog_ref'],
            'selection':common['feature_selection'],'training_sessions':training_dates,'prediction_sessions':prediction_dates,
            'parents_by_session':parents,'rows':rows}
        features={**features,'feature_ref':digest(features)}
        label_summary={'contract_version':'stock_fold_label_slice_v3','input_manifest_ref':digest(inputs),
            'prepared_view_ref':view['prepared_view_ref'],'fold_spec_ref':fold_ref,'cutoff':spec['fit_cutoff'],
            'selectors':{k:deepcopy(inputs['selectors'][k]) for k in ('training_labels','evaluation_labels')},
            'core_result_refs':list(inputs['core_result_refs']),'raw_label_refs':sorted(self.raw_refs[fold_ref]),
            'training_sessions':training_dates,'training_row_count':len(train),'excluded':excluded}
        labels={**label_summary,'label_ref':digest(label_summary)}
        evaluation_rows=[_target_row(self.targets,'evaluation_raw_labels',fold_ref,off,key) for off,key in
                         zip(selected['evaluation_labels'],self.feature.keys(selected['evaluation_labels']))]
        ev={'contract_version':'stock_matrix_evaluation_target_slice_v1','prepared_view_ref':view['prepared_view_ref'],
            'fold_spec_ref':fold_ref,'selector':deepcopy(inputs['selectors']['evaluation_labels']),
            'cutoff':spec['evaluation_cutoff'],'rows':evaluation_rows}
        evaluation={**ev,'label_ref':digest(ev)}
        def training_rows():
            for i,(off,key) in enumerate(zip(train,keys)):
                block=next(b for b in self.feature._blocks if b['start']<=int(off)<b['start']+b['count'])
                row=block['rows'].row(int(off)-block['start'],security=key[0],session=key[1],
                                     values=[float(v) for v in X[i]],validity=[True]*len(columns))
                target=_target_row(self.targets,'training_normalized_labels',fold_ref,off,key)
                row.update(label=float(y[i]),raw_return=target['return'],label_available_at=target['label_available_at'],
                           normalized_available_at=target['normalized_available_at'])
                yield row
        training_rows_ref=_stream_ref(training_rows())
        lease_bytes=X.nbytes+y.nbytes+P.nbytes+_resident_size([features,labels,common,keys,candidate_keys,evaluation])
        require(self.store.resident_bytes+self.store.lease_bytes+lease_bytes<=self.store.maximum_matrix_bytes,
                'active fold resident byte budget exceeded')
        self.store.lease_bytes+=lease_bytes; self.store.metrics['fold_projection_calls']+=1
        self.store.check()
        return MatrixFoldProjection(self.store,X=X,y=y,P=P,features=features,labels=labels,common=common,
            feature_rows=rows,feature_ref=features['feature_ref'],label_ref=labels['label_ref'],
            training_keys=keys,candidate_keys=candidate_keys,training_rows_ref=training_rows_ref,
            excluded=excluded,raw_refs=sorted(self.raw_refs[fold_ref]),evaluation=evaluation,
            _lease_bytes=lease_bytes)


def _admit_matrix_view(descriptor,folds,*,limits=None,_store=None):
    budgets={'maximum_source_bytes':8*1024**3,'maximum_matrix_bytes':512*1024**2} if limits is None else deepcopy(limits)
    _fields(budgets,{'maximum_source_bytes','maximum_matrix_bytes'},'exact matrix batch byte limits required')
    store=_store or VerifiedMatrixStore(**budgets)
    try:
        view=store.read_json(descriptor,'prepared_view_ref')
        _fields(view,{'contract_version','definition','definition_ref','schema','schema_digest','row_index','source_selection',
                      'partitions','core_results','prepared_view_ref'},'exact prepared matrix view required')
        require(view['contract_version']=='stock_ml_prepared_view_v1' and view['definition_ref']==digest(view['definition']) and
                view['schema_digest']==digest(view['schema']),'prepared matrix definition/schema mismatch')
        definition=view['definition']; common_keys=('scope','snapshot','pit_policy','calendar','universe','catalog_ref','feature_selection','ordered_features')
        _fields(definition,set(common_keys)|{'feature_inputs','fold_specs','preparation_options','implementation_sources',
                                         'implementation_ref','environment'},'exact prepared matrix definition required')
        require(definition['implementation_ref']==digest(definition['implementation_sources']),'prepared matrix implementation mismatch')
        _fields(view['schema'],{'features','training_raw_labels','training_normalized_labels','evaluation_raw_labels'},'exact prepared table schema required')
        _schema(view['schema']['features'],definition['ordered_features'])
        for table in ('training_raw_labels','training_normalized_labels','evaluation_raw_labels'):
            names=_schema(view['schema'][table]); require(len(names)==1,'one-column Target table required')
        row_index=_row_index(store.read_json(view['row_index'],'row_index_ref'))
        feature_desc=definition['feature_inputs']; _fields(feature_desc,{'path','file_digest','feature_inputs_ref'},'explicit Feature index descriptor required')
        # Feature identity has its own projection formula; do not apply the
        # generic self-exclusion ref rule to its saved index.
        store._register(feature_desc); index=store._decode_once(feature_desc['path'])
        store.json[feature_desc['path']]=index
        require(index['feature_inputs_ref']==feature_desc['feature_inputs_ref'],'prepared Feature index identity mismatch')
        if index['contract_version']=='stock_feature_inputs_v2':
            feature=load_feature_matrix_index(Path(feature_desc['path']).parent,store=store,_index=index)
            require([p for p in view['partitions'] if p['table']=='features']==index['partitions'],
                    'prepared Feature partitions changed from frozen index')
        else:
            from .stock_feature_inputs import _spec, _qlib, _qlib_scope, INDEX_FIELDS
            require(index['contract_version']=='stock_feature_inputs_v1','unsupported prepared Feature ancestry')
            _fields(index,INDEX_FIELDS,'exact original Feature input index required'); _verify_ref(index,'content_digest')
            definition1=index['definition']; _fields(definition1,{'spec','shard_sessions','implementation_sources','implementation_ref','environment'},
                'exact original Feature input definition required')
            require(index['status']=='COMPLETE' and definition1['implementation_ref']==digest(definition1['implementation_sources']) and
                index['definition_ref']==digest(definition1) and type(definition1['shard_sessions']) is int and definition1['shard_sessions']>0 and
                index['feature_inputs_ref']==digest({k:index[k] for k in ('definition_ref','qlib_manifest','feature_parents')}),
                'original Feature input identity/definition mismatch')
            spec=index['definition']['spec']; universe_id=_spec(spec,reader=store.read_json)
            qview=_qlib(index['qlib_manifest'],{'maximum_parent_bytes':store.maximum_parent_bytes},fingerprints=store.fingerprints,
                _file_hasher=store._hash_once,_json_reader=store._decode_once)
            require(qview==index['qlib_view'],'original Feature Qlib identity mismatch'); _qlib_scope(qview,spec,universe_id)
            partitions=[p for p in view['partitions'] if p['table']=='features']
            covered,blocks,hashes,parents=admit_feature_matrix_parts(store,spec=spec,view=index['qlib_view'],
                schema=view['schema']['features'],row_index=row_index,partitions=partitions,complete=True,universe_id=universe_id)
            original_blocks={(tuple(b['_sessions']),b['_original_feature_ref']) for b in blocks}
            parent_dates=[]
            for parent in index['feature_parents']:
                _fields(parent,{'features','input_evidence','sessions'},'exact original Feature parent required')
                require(parent['sessions']==spec['feature_sessions'][len(parent_dates):len(parent_dates)+len(parent['sessions'])] and
                    (tuple(parent['sessions']),parent['features']['feature_ref']) in original_blocks,
                    'converted matrix differs from original Feature shard')
                old=store.read_json(parent['features'],'feature_ref')
                proof_desc=parent['input_evidence']; _fields(proof_desc,{'path','file_digest','input_evidence_ref'},'original Feature proof descriptor required')
                store._register(proof_desc); proof=store._decode_once(proof_desc['path'])
                require(digest(proof)==proof_desc['input_evidence_ref']==old['input_evidence_ref'] and
                        old['contract_version']=='stock_feature_build_v1' and old['catalog_ref']==spec['catalog_ref'] and
                        old['selection']==spec['feature_selection'] and old['ordered_features']==spec['ordered_features'] and
                        old['qlib_view']==index['qlib_view'],'original Feature converted proof/schema mismatch')
                parent_dates.extend(parent['sessions'])
                store.drop_json(parent['features'],'feature_ref'); store._decoded.pop(proof_desc['path'],None)
                del old,proof
            require(parent_dates==spec['feature_sessions'] and len(original_blocks)==len(index['feature_parents']),
                    'original Feature shard coverage mismatch')
            feature=FeatureMatrixInputs(_TOKEN,path=Path(feature_desc['path']).parent,index=index,store=store,row_index=row_index,blocks=blocks)
            feature._source_hashes=hashes; feature._parents_by_session=parents; store.metrics['common_key_index_builds']+=1
        require(feature._row_index==row_index and row_index['security_ids']==definition['universe'],'prepared matrix Feature row scope mismatch')
        spec=feature._index['definition']['spec']
        require(all(definition[k]==spec[k] for k in common_keys),'prepared matrix common source mismatch')
        fold_specs={}
        previous=None
        for spec in definition['fold_specs']:
            validate_spec(spec,definition['calendar']); ref=digest(spec)
            require(ref not in fold_specs and (previous is None or spec['oos_trade_sessions'][0]>previous),
                    'ordered disjoint prepared folds required')
            fold_specs[ref]=spec; previous=spec['oos_trade_sessions'][-1]
        require(bool(fold_specs),'prepared matrix folds required')
        wrappers={}
        require(type(view['core_results']) is list and bool(view['core_results']),'prepared saved Core results required')
        for desc in view['core_results']:
            wrapper=store.read_json(desc,'core_result_artifact_ref'); ref=wrapper['result']['metadata']['result_ref']
            require(ref not in wrappers,'duplicate prepared Core result'); wrappers[ref]=wrapper
        targets,hashes,raw_refs,label_selection=_label_admission([p for p in view['partitions'] if p['table']!='features'],
            view=view,feature=feature,store=store,row_index=row_index,core_wrappers=wrappers,common=definition,fold_specs=fold_specs)
        selection=store.read_json(view['source_selection'],'source_selection_ref')
        _fields(selection,{'contract_version','feature_inputs_ref','feature_rows','label_rows','source_selection_ref'},'exact prepared source-selection required')
        require(selection['contract_version']=='stock_matrix_source_selection_v1' and selection['feature_inputs_ref']==feature.identity and
                selection['feature_rows']==[r for b in feature._blocks for r in b['_source_selection_rows']],
                'prepared Feature source-selection mismatch')
        require(len(selection['label_rows'])==len(label_selection) and sorted(digest(r) for r in selection['label_rows'])==
                sorted(digest(r) for r in label_selection),'prepared actual Target source-selection mismatch')
        selectors={}
        for inputs,spec in folds:
            require(digest(spec) in fold_specs and spec==fold_specs[digest(spec)],'saved fold outside prepared definition')
            _saved_inputs(inputs,spec,view); selected={}; entries={}
            for role,desc in inputs['selectors'].items():
                if desc is None: continue
                selected[role],entries[role]=_selector(desc,role=role,inputs=inputs,spec=spec,view=view,row_index=row_index,store=store)
            require(bool(store.np.array_equal(selected['training'],selected['training_labels'])),'training X/y selectors must have identical keys')
            training_dates,inference_dates=validate_spec(spec,definition['calendar'])
            expected=feature.offsets([[s,d] for d in inference_dates for s in definition['universe']])
            require(bool(store.np.array_equal(selected['inference'],expected)) and
                    bool(store.np.array_equal(selected['evaluation_labels'],expected)),'complete OOS grid selectors required')
            require(all(key[1] in training_dates for key in feature.keys(selected['training'])),'training selector outside declared window')
            require(all(ref in wrappers for ref in inputs['core_result_refs']),'fold Core refs outside prepared view')
            required_core=[]
            for block in sorted(targets.get((digest(spec),'training_normalized_labels'),[]),key=lambda b:b.start):
                for ref in block.core_refs:
                    if ref not in required_core: required_core.append(ref)
            require(inputs['core_result_refs']==required_core,'fold Core result refs do not match its admitted targets')
            for ref in inputs['core_result_refs']:
                result=wrappers[ref]['result']
                require(set(result['sessions'])<=set(training_dates),'fold Core result outside training window')
            selectors[digest(inputs),digest(spec)]=selected
        store.metrics['resident_metadata_bytes']=store.resident_bytes; store.check()
        return MatrixBatchState(_TOKEN,store=store,view=view,feature=feature,row_index=row_index,targets=targets,
                                raw_refs=raw_refs,core_wrappers=wrappers,selectors=selectors)
    except BaseException:
        for array in store.arrays.values():
            if hasattr(array,'_mmap'): array._mmap.close()
        raise


def load_matrix_batch_state(manifest,*,limits=None,_store=None):
    _fields(manifest,{'contract_version','definition','definition_ref','prepared_view','folds','status','batch_ref','content_digest'},
            'exact matrix batch manifest required')
    require(manifest['contract_version']=='stock_ml_batch_inputs_v2' and manifest['status']=='COMPLETE','complete matrix batch required')
    _verify_ref(manifest,'content_digest')
    require(manifest['batch_ref']==digest({k:v for k,v in manifest.items() if k not in ('batch_ref','content_digest')}), 'matrix batch ref mismatch')
    definition=manifest['definition']; _fields(definition,{'version','feature_inputs','fold_specs','preparation_options',
        'implementation_sources','implementation_ref','environment'},'exact matrix batch definition required')
    require(definition['version']=='axiom.stock_ml_batch_inputs/2' and manifest['definition_ref']==digest(definition) and
            definition['implementation_ref']==digest(definition['implementation_sources']),'matrix batch definition/implementation mismatch')
    options=definition['preparation_options']; _fields(options,{'row_block_sessions','column_block','maximum_resident_bytes','normalization_backend'},'exact preparation options required')
    require(options['normalization_backend']=='core_cs_batch_v1' and all(type(options[k]) is int and options[k]>0 for k in options if k!='normalization_backend'),
            'positive preparation budgets required')
    require(type(manifest['folds']) is list and bool(manifest['folds']),'matrix batch folds required')
    folds=[]
    for f in manifest['folds']:
        _fields(f,{'input_manifest','fold_spec'},'explicit matrix batch fold required')
        require(f['input_manifest']['prepared_view']==manifest['prepared_view'],'matrix batch fold view mismatch')
        folds.append((f['input_manifest'],f['fold_spec']))
    require([s for _,s in folds]==definition['fold_specs'],'matrix batch original fold specs mismatch')
    state=_admit_matrix_view(manifest['prepared_view'],folds,limits=limits,_store=_store)
    if not all(state.view['definition'][k]==definition[k] for k in definition if k!='version'):
        state.close(); raise ValueError('matrix batch/prepared definition mismatch')
    return state


def load_matrix_input_projection(inputs,spec,*,limits=None):
    """Independent saved-fold admission, without inventing a persisted batch."""
    state=_admit_matrix_view(inputs['prepared_view'],[(inputs,spec)],limits=limits)
    try:
        result=state.project(inputs,spec); result._owned_state=state; return result
    except BaseException:
        state.close(); raise


def _validate_staged_matrix_batch(manifest,*,stage,target):
    """Validate a complete staging closure before its one atomic publication.

    Saved identities retain the final canonical paths. Only the internal file
    locator redirects paths under this exact target to its staging directory;
    the same admission performs every byte/ref/schema/clock check.
    """
    stage=Path(stage).resolve(); target=Path(target).resolve()
    require(stage.is_dir() and not target.exists() and stage!=target,'fresh bounded matrix stage required')
    def locate(path):
        path=Path(path)
        try: relative=path.relative_to(target)
        except ValueError: return path
        physical=stage/relative
        require(physical.resolve().is_relative_to(stage),'staged matrix file escaped its directory')
        return physical
    store=VerifiedMatrixStore(_path_resolver=locate)
    state=load_matrix_batch_state(manifest,_store=store)
    try: state.store.check(); return deepcopy(state.store.metrics)
    finally: state.close()

"""Compact reader evidence using the existing immutable buffer/JSON codecs.

Cell selection, clocks and missing states remain exact. Shared coverage and
report/interval components are stored once, and repeated field rows share a
chunk-local template. Legacy evidence stays readable without rewriting it.
"""
import json
import shutil
import zlib
from hashlib import sha256
from pathlib import Path

from .stock_artifacts import digest, file_digest
from .stock_fold_inputs import require, seal

_MINIMUM_FREE_BYTES = 0
_SHARED_KEYS = {'coverage', 'components', 'selected_intervals'}


def configure_input_disk_reserve(minimum_free_bytes):
    global _MINIMUM_FREE_BYTES
    require(type(minimum_free_bytes) is int and minimum_free_bytes >= 0, 'invalid input disk reserve')
    _MINIMUM_FREE_BYTES = minimum_free_bytes


def check_input_disk(root, pending_bytes=0):
    if _MINIMUM_FREE_BYTES:
        # Writers may check a destination before creating its directory.
        location=Path(root).resolve()
        while not location.exists():
            location=location.parent
        if shutil.disk_usage(location).free < _MINIMUM_FREE_BYTES + pending_bytes:
            raise RuntimeError('DISK_RESERVE_REACHED: committed input checkpoints retained')


def compact_reader_evidence(root, child):
    from .stock_matrix_storage import write_buffer
    import numpy as np
    shared_root=Path(root).parent/'.shared-input-evidence'
    shared_root.mkdir(parents=True,exist_ok=True)
    objects={}
    def pack(value):
        if type(value) is list: return [pack(v) for v in value]
        if type(value) is not dict: return value
        result={}
        for key,item in value.items():
            if key in _SHARED_KEYS and type(item) in (dict,list) and item:
                raw=json.dumps(item,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
                ref='sha256:'+sha256(raw).hexdigest()
                if ref not in objects:
                    compressed=zlib.compress(raw)
                    descriptor=write_buffer(shared_root,np.frombuffer(compressed,dtype='u1'),dtype='uint8',shape=[len(compressed)])
                    objects[ref]={'buffer':descriptor,'value_ref':ref,'decoded_bytes':len(raw),'encoding':'zlib_json_v1'}
                descriptor=objects[ref]
                result[key]={'shared_evidence':descriptor}
            else: result[key]=pack(item)
        return result
    context=pack(child['context']); metadata={}; templates={}
    for field,meta in child['field_meta'].items():
        rows=[]
        for row in meta['by_key']:
            template=pack({k:v for k,v in row.items() if k not in ('security_id','session')})
            ref=digest(template);templates.setdefault(ref,template)
            rows.append({'security_id':row['security_id'],'session':row['session'],'shared_row_ref':ref})
        metadata[field]={**{k:pack(v) for k,v in meta.items() if k!='by_key'},'by_key':rows}
    return seal({'contract_version':'reader_column_evidence_v2','context':context,
        'field_meta':metadata,'fields':child['fields'],'sessions':child['sessions'],
        'shared_rows':templates},'evidence_ref')


def shared_evidence_descriptors(value):
    if type(value) is dict:
        if set(value)=={'shared_evidence'}:
            item=value['shared_evidence']
            require(set(item)=={'buffer','value_ref','decoded_bytes','encoding'} and
                    item['encoding']=='zlib_json_v1' and type(item['decoded_bytes']) is int and item['decoded_bytes']>0,
                    'invalid shared input evidence descriptor')
            yield item
        else:
            for v in value.values(): yield from shared_evidence_descriptors(v)
    elif type(value) is list:
        for v in value: yield from shared_evidence_descriptors(v)


def resolve_evidence_value(value, *, cache=None, maximum_decoded_bytes=512*1024**2):
    """Resolve exact source proof on demand; ordinary admission leaves coverage referenced."""
    cache={} if cache is None else cache
    if type(value) is dict:
        if set(value)=={'shared_evidence'}:
            item=value['shared_evidence'];ref=item['value_ref'];desc=item['buffer']
            if ref not in cache:
                require(item['decoded_bytes']<=maximum_decoded_bytes,'shared input evidence decode budget exceeded')
                path=Path(desc['path'])
                require(not path.is_symlink() and path.stat().st_size==desc['shape'][0] and
                        file_digest(path)==desc['file_digest']==desc['buffer_digest'],'shared input evidence changed')
                decoder=zlib.decompressobj();raw=decoder.decompress(path.read_bytes(),item['decoded_bytes']+1)
                require(decoder.eof and not decoder.unused_data and len(raw)==item['decoded_bytes'] and
                        'sha256:'+sha256(raw).hexdigest()==ref,'shared input evidence content mismatch')
                cache[ref]=json.loads(raw)
            return cache[ref]
        return {k:resolve_evidence_value(v,cache=cache,maximum_decoded_bytes=maximum_decoded_bytes) for k,v in value.items()}
    if type(value) is list: return [resolve_evidence_value(v,cache=cache,maximum_decoded_bytes=maximum_decoded_bytes) for v in value]
    return value


def materialize_evidence_rows(child):
    if child['contract_version']=='reader_column_evidence_v1': return child
    templates=child['shared_rows'];cache={};resolved={}
    for ref,row in templates.items():
        require(digest(row)==ref and not {'security_id','session'}&set(row),'shared row template mismatch')
        resolved[ref]=resolve_evidence_value(row,cache=cache)
    metadata={}
    for field,meta in child['field_meta'].items():
        rows=[]
        for row in meta['by_key']:
            require(set(row)=={'security_id','session','shared_row_ref'} and row['shared_row_ref'] in resolved,'shared row reference missing')
            rows.append({**resolved[row['shared_row_ref']],'security_id':row['security_id'],'session':row['session']})
        metadata[field]={**meta,'by_key':rows}
    return {**child,'field_meta':metadata}


BLOCK_VERSION='reader_column_block_v1'


def input_block_receipt(*, spec, sessions, fields, context, definitions, bindings=None):
    """Small production receipt, not a reversible encoding of per-cell proofs.

    Published source binds this receipt and immutable numeric buffers together.
    Original facts remain reproducible from the pinned source and recipe. No
    coverage graph, component list, daily row template or selected value lives here.
    """
    from .stock_artifacts import digest
    context={k:context[k] for k in ('domain','snapshot_id','source_profile_id',
        'contract_id','reader_version','event_reader_version','selection',
        'benchmark_security_id','qualification') if k in context}
    require(context.get('snapshot_id') not in (None,'','current','latest'),'pinned input receipt source required')
    metadata={field:{'dtype':definitions[field]['dtype'],'unit':definitions[field].get('unit'),
        'source_binding':(bindings[field] if bindings is not None else digest({
            'context':context,'field':field,'policy':spec['pit_policy']}))} for field in fields}
    return seal({'contract_version':BLOCK_VERSION,'context':context,'fields':list(fields),
        'sessions':list(sessions),'security_axis_ref':digest(spec['universe']),
        'field_meta':metadata},'evidence_ref')

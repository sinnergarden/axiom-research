"""Saved packed-Core Feature proof; hash admission and codecs, no execution."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os

from .stock_artifacts import digest, file_digest, _read, _verify_ref
from .stock_fold_inputs import require, seal, ordered, grid, file_fingerprint
from .stock_label_contracts import _finite, _instant, _session
from .stock_matrix_storage import DTYPES, _buffer_array, instant_us, write_buffer, write_part, write_partition
from .stock_matrix_feature_sources import BUFFER_TYPES, _fields, _ref, _row_buffers, _output

METADATA_VERSION = 'stock_matrix_feature_metadata_v3'
METADATA_FIELDS = {'contract_version','sessions','ordered_features','catalog_ref','selection',
    'qlib_view','schema','rows','row_references','source_descriptor','input_evidence','day_outputs','metadata_ref'}
ROW_FIELDS = {'security_id','session','values','validity','availability','reasons',
              'member','knowledge_cutoff','source_refs'}
SELECTION_FIELDS = {'source_ref','session','history_sessions','anchor_session','knowledge_cutoff'}
SOURCE_BUFFERS = {'values':'float64_le','original_values':'float64_le','validity':'bool_u8',
    'available':'int64_le','available_validity':'bool_u8','reasons':'uint8',
    'member':'bool_u8','member_available':'int64_le','member_available_validity':'bool_u8'}


def _core_ref(value):
    # Core's packed contract uses JSON's default ASCII escaping, deliberately
    # distinct from Research's canonical document encoder.
    return 'sha256:'+sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _validate_core_header(header, rows, selection, spec, schema):
    require(header.get('contract_version')=='feature_column_result_v1' and
        not any(k in header for k in ('values','validity','reason_codes','available_at_utc_us','source_codes','stats')),
        'packed Core binding header required, without arrays or diagnostics')
    require(header['result_ref']==_core_ref({k:v for k,v in header.items() if k!='result_ref'}),
        'packed Core result identity mismatch')
    require(header['input_ref']==_core_ref({'metadata':header['input_metadata'],'buffers':header['input_buffers']}),
        'packed Core input identity mismatch')
    refs={'source_ref':selection['source_ref'],'selection_ref':digest(selection)}
    require(header['input_refs']==refs and header['input_metadata']['input_refs']==refs and
        header['input_metadata']['source_bindings']==header['source_bindings'] and
        header['input_metadata']['sessions']==selection['history_sessions'] and
        header['input_metadata']['security_ids']==spec['universe'], 'packed Core input selection refs/axes mismatch')
    require(header['sessions']==[selection['session']] and header['security_ids']==spec['universe'] and
        header['null_available_at_utc_us']==-(2**63), 'packed Core output axes/null clock mismatch')
    packed=_row_buffers(rows,len(schema))
    clocks=_buffer_array((instant_us(a) if a is not None else header['null_available_at_utc_us']
        for row in rows for a in row['availability']),'int64_le')
    dictionary=header['reason_dictionary']
    require(type(dictionary) is list and dictionary and dictionary[0] is None and
        len(set(dictionary))==len(dictionary),'packed Core reason dictionary mismatch')
    reason_ids={reason:i for i,reason in enumerate(dictionary)}
    require(all(len(reasons)<=1 and (not reasons or reasons[0] in reason_ids and reason_ids[reasons[0]]>0)
        for row in rows for reasons in row['reasons']),'packed Core single reason required')
    reasons=_buffer_array((reason_ids[r[0]] if r else 0 for row in rows for r in row['reasons']),'int32_le')
    for name,value,dtype in [('values',packed['values'],'<f8'),
        ('validity',packed['value_validity'],'|b1'),('available_at_utc_us',clocks,'<i8'),
        ('reason_codes',reasons,'<i4')]:
        require(header['output_buffers'][name]=={'shape':[1,len(rows),len(schema)],'dtype':dtype,
            'sha256':sha256(memoryview(value).cast('B')).hexdigest()},'packed Core output buffer differs from saved rows')


def _physical(desc, store):
    require(type(desc.get('path')) is str and Path(desc['path']).is_absolute() and _ref(desc.get('file_digest')),
            'fixed column source descriptor required')
    physical=Path(desc['path']) if store is None else store._resolve(desc['path'])
    require(not physical.is_symlink(),'fixed column source file required')
    if store is not None:
        store._register(desc)
    else:
        mark=file_fingerprint(desc['path'])
        require(file_digest(desc['path'])==desc['file_digest'] and
                file_fingerprint(desc['path'])==mark,'column source file changed')


def _saved_json(desc, ref_key, store):
    if store is not None and ref_key!='evidence_ref':
        return store.read_json(desc,ref_key,_revisit=True)
    path=Path(desc['path']) if store is None else store._resolve(desc['path'])
    mark=file_fingerprint(path)
    require(mark[2]<=min(64*1024**2,store.maximum_parent_bytes if store is not None else 64*1024**2),
            'column source JSON byte budget exceeded')
    if store is not None:
        require(mark==store.fingerprints[desc['path']],'column source JSON changed before decode')
        require(store._native_caller_bytes()+16*mark[2]<=store.maximum_matrix_bytes,
                'column source JSON decode workspace budget exceeded')
    def opened_mark(stream):
        stat=os.fstat(stream.fileno())
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns
    with path.open('rb') as stream:
        require(opened_mark(stream)==mark,'column source JSON inode changed before decode')
        if store is not None: store.metrics['json_decode_calls']+=1
        value=_read(path,_stream=stream,_expected_bytes=mark[2],_expected_digest=desc['file_digest'])
        require(opened_mark(stream)==mark and file_fingerprint(path)==mark,
                'column source JSON changed during read')
    return value


def verify_feature_column_source(descriptor, *, store=None, _dependency_receipts=None,
                                 _evidence_days=None, _evidence_fields=None,
                                 _verify_buffers=True):
    """Verify JSON and every physical leaf without loading any source arrays.

    A supplied matrix Store owns hash reuse and change detection. The ordinary
    Feature owner narrows evidence to requested leaves; omitted selectors keep
    the explicit deep-audit behavior. Deferred buffers are checked by the same
    Store on first use, never accepted through a persisted trusted flag.
    """
    _fields(descriptor,{'path','file_digest','source_ref'},'exact column source descriptor required')
    _physical(descriptor,store)
    manifest=_saved_json(descriptor,'source_ref',store)
    _verify_ref(manifest,'source_ref')
    require(manifest['source_ref']==descriptor['source_ref'] and
            manifest['contract_version']=='research_feature_column_source_v1', 'column source identity/version mismatch')
    require(manifest['implementation_ref']==digest(manifest['implementation_sources']),
            'column source implementation binding mismatch')
    from .stock_feature_columns import REASONS
    reasons=manifest['reason_codes']
    require(type(reasons) is list and 0<len(reasons)<=256 and len(reasons)==len(set(reasons)) and
            reasons[:len(REASONS)]==list(REASONS[:len(reasons)]) and
            all(isinstance(r,str) and bool(r) for r in reasons[1:]),'input reason dictionary mismatch')
    spec=manifest['spec']; days=ordered(spec['read_sessions'],'column source dates')
    securities=ordered(spec['universe'],'column source securities')
    for day in days: _session(day)
    require(set(ordered(spec['feature_sessions'],'column source output dates'))<=set(days),
            'column source output dates outside physical calendar')
    require(spec['pit_policy']=='best_effort_vendor_v1' and manifest['selection_invariance']==
        'declared_vendor_release_visible_at_each_historical_row_cutoff','unsupported column source selection policy')
    require(set(spec['cutoff_by_session'])>=set(days) and
            all(_instant(spec['cutoff_by_session'][a])<=_instant(spec['cutoff_by_session'][b])
                for a,b in zip(days,days[1:])), 'column source cutoff order mismatch')
    financial=manifest.get('financial_materializations',{})
    if financial or any(m.get('domain')=='materialized_financial_events'
                        for m in manifest.get('daily_fields',{}).values()):
        from .financial_feature_columns import validate_financial_materializations
        validate_financial_materializations(manifest)
    extra=manifest.get('daily_fields',{})
    require(type(extra) is dict and all(type(f) is str and f.isidentifier() and
        set(m)=={'domain','dtype','unit'} and m['dtype'] in ('float64','int64','bool') for f,m in extra.items()),
        'typed daily fields required')
    buffer_types=dict(SOURCE_BUFFERS)
    for name,dtype in {'member_known':'bool_u8','member_pool_complete':'bool_u8','member_reasons':'uint8'}.items():
        if name in manifest['buffers']: buffer_types[name]=dtype
    for field in extra:
        require(field not in ('open','high','low','close','amount_cny','factor','is_member','vwap'),
                'daily field shadows existing source')
        buffer_types.update({field+'::'+k:v for k,v in SOURCE_BUFFERS.items() if not k.startswith('member')})
    require(type(manifest['buffers']) is dict and set(manifest['buffers'])==set(buffer_types),
            'column source buffers mismatch')
    reachable={manifest['source_ref'],descriptor['file_digest']}
    for name,desc in manifest['buffers'].items():
        _fields(desc,{'path','file_digest','dtype','shape','buffer_digest'},'exact source buffer required')
        shape=[len(days),len(securities)]+([] if name.startswith('member') or '::' in name else [6])
        require(desc['dtype']==buffer_types[name] and desc['shape']==shape and
                desc['buffer_digest']==desc['file_digest'],'column source buffer shape/type/ref mismatch')
        if _verify_buffers: _physical(desc,store)
        size=DTYPES[desc['dtype']][1]
        for dimension in shape: size*=dimension
        physical=Path(desc['path']) if store is None else store._resolve(desc['path'])
        require(physical.stat().st_size==size,'column source buffer length mismatch')
        reachable.add(desc['buffer_digest'])
    require(type(manifest['chunks']) is list and bool(manifest['chunks']),'column source provenance chunks required')
    covered=[]
    all_chunks=[(chunk,3,None) for chunk in manifest['chunks']]
    extension_fields=[]
    for extension in manifest.get('daily_extensions',[]):
        require(set(extension)=={'fields','chunks'} and bool(extension['fields']), 'daily extension closure required')
        extension_fields.extend(extension['fields'])
        require([d for c in extension['chunks'] for d in c['sessions']]==days, 'daily extension calendar coverage mismatch')
        all_chunks.extend((c,None,extension['fields']) for c in extension['chunks'])
    require(len(extension_fields)==len(set(extension_fields)) and set(extension_fields)==set(extra),
            'daily extension field coverage mismatch')
    for chunk,expected_domains,expected_fields in all_chunks:
        _fields(chunk,{'sessions','evidence'},'exact column source chunk required')
        chunk_days=ordered(chunk['sessions'],'source chunk dates')
        if expected_domains is not None: covered.extend(chunk_days)
        require(type(chunk['evidence']) is list and bool(chunk['evidence']) and
            (expected_domains is None or len(chunk['evidence'])==expected_domains),'source domains required')
        if _evidence_days is not None and not set(chunk_days).intersection(_evidence_days):
            continue
        chunk_fields=set(expected_fields) if expected_fields is not None else {
            'open','high','low','close','amount_cny','factor','is_member'}
        if _evidence_fields is not None and not chunk_fields.intersection(_evidence_fields):
            continue
        seen_fields=[]
        for desc in chunk['evidence']:
            _fields(desc,{'path','file_digest','evidence_ref'},'exact source provenance descriptor required')
            _physical(desc,store)
            token=('feature_column_evidence_checked_v1',desc['path'],desc['file_digest'],desc['evidence_ref'],
                   None if expected_fields is None else tuple(expected_fields))
            known=None if store is None else store.content.get(token)
            if known is None:
                child=_saved_json(desc,'evidence_ref',store)
                _verify_ref(child,'evidence_ref')
                light=child['contract_version']=='reader_column_block_v1'
                compact=child['contract_version']=='reader_column_evidence_v2'
                _fields(child,{'contract_version','context','field_meta','fields','sessions','evidence_ref'} |
                    ({'shared_rows'} if compact else {'security_axis_ref'} if light else set()),
                    'exact column source provenance required')
                require(child['contract_version'] in ('reader_column_evidence_v1','reader_column_evidence_v2','reader_column_block_v1') and
                        child['sessions']==chunk_days and child['evidence_ref']==desc['evidence_ref'],
                        'column source provenance closure mismatch')
                if light:
                    require(child['security_axis_ref']==digest(securities) and
                            set(child['fields'])==set(child['field_meta']),'input block axes/fields mismatch')
                    require(child['context'].get('snapshot_id') not in (None,'','current','latest'),
                            'pinned input block source required')
                    for field,metadata in child['field_meta'].items():
                        _fields(metadata,{'dtype','unit','source_binding'},'input block metadata required')
                        require(_ref(metadata['source_binding']),'input block source version required')
                shared_objects={}
                if compact:
                    from .feature_input_evidence import shared_evidence_descriptors, materialize_evidence_rows
                    for item in shared_evidence_descriptors(child):
                        proof=item['buffer']
                        _fields(proof,{'path','file_digest','dtype','shape','buffer_digest'},'exact shared evidence buffer required')
                        require(proof['dtype']=='uint8' and len(proof['shape'])==1 and proof['buffer_digest']==proof['file_digest'], 'shared evidence buffer mismatch')
                        if proof['path'] in shared_objects: continue
                        shared_objects[proof['path']]=proof
                        _physical(proof,store)
                        require(Path(proof['path']).stat().st_size==proof['shape'][0], 'shared evidence buffer size mismatch')
                        reachable.update((proof['file_digest'],item['value_ref']))
                    child=materialize_evidence_rows(child)
                if expected_fields is not None:
                    require(set(child['fields'])==set(child['field_meta']) and
                            set(child['fields'])<=set(expected_fields), 'daily evidence fields mismatch')
                    for field,meta in child['field_meta'].items():
                        declaration=extra[field]
                        require(child['context']['domain']==declaration['domain'] and
                                meta['dtype']==declaration['dtype'] and meta['unit']==declaration['unit'],
                                'daily evidence domain/type/unit mismatch')
                        if not light:
                            keys=[(r['security_id'],r['session']) for r in meta['by_key']]
                            covered_securities=financial[field]['coverage_security_ids'] if field in financial else securities
                            require(len(keys)==len(set(keys)) and set(keys)=={(s,d) for d in chunk_days for s in covered_securities},
                                    'daily evidence key coverage mismatch')
                    seen_fields.extend(child['fields'])
                if financial and not light:
                    validate_financial_materializations(manifest,child=child)
                if _dependency_receipts is not None:
                    _dependency_receipts(child)
                if store is not None:
                    store.drop_json(desc,'evidence_ref')
                    # The existing Store charges this small receipt; it still
                    # fingerprints every referenced file on each admission.
                    store._retain_entry(store.content,token,(tuple(chunk_days),tuple(child['fields']),tuple(shared_objects.values())))
                del child
            else:
                require(known[0]==tuple(chunk_days),'cached source provenance date mismatch')
                for proof in known[2]:
                    _physical(proof,store)
                    reachable.add(proof['file_digest'])
                if expected_fields is not None: seen_fields.extend(known[1])
            reachable.update((desc['evidence_ref'],desc['file_digest']))
        if expected_fields is not None:
            require(len(seen_fields)==len(set(seen_fields)) and set(seen_fields)==set(expected_fields),
                    'daily evidence field closure mismatch')
    require(covered==days,'column source provenance calendar coverage mismatch')
    return manifest,reachable


def _day_ref(proof, output, spec, view, schema):
    return digest({'contract_version':'stock_feature_day_v3','catalog_ref':spec['catalog_ref'],
        'selection':spec['feature_selection'],'ordered_features':spec['ordered_features'],
        'schema_ref':digest(schema),'qlib_view_ref':digest(view),'selection_ref':proof['selection_ref'],
        'core_header_ref':digest(proof['core_header']),'output_ref':output['output_ref']})


def validate_source_projection(spec, source_spec):
    """A target may select a shorter interval from the same frozen source."""
    require(all(spec[key]==source_spec[key] for key in
        ('calendar','universe','snapshot','pit_policy','scope')),
        'column Feature target/source frozen scope mismatch')
    require(set(spec['read_sessions'])<=set(source_spec['read_sessions']) and
        set(spec['feature_sessions'])<=set(source_spec['feature_sessions']) and
        all(_instant(spec['cutoff_by_session'][day])==_instant(source_spec['cutoff_by_session'][day])
            for day in spec['read_sessions']), 'column Feature target/source date/cutoff projection mismatch')


def validate_metadata(metadata, rows, spec, view, source_manifest, outputs, *, universe_id):
    _fields(metadata,METADATA_FIELDS,'exact column Feature metadata required')
    _verify_ref(metadata,'metadata_ref')
    require(metadata['contract_version']==METADATA_VERSION,'column Feature metadata version mismatch')
    days=ordered(metadata['sessions'],'column Feature sessions'); width=len(spec['ordered_features'])
    require(metadata['ordered_features']==spec['ordered_features'] and metadata['catalog_ref']==spec['catalog_ref'] and
        metadata['selection']==spec['feature_selection'] and metadata['qlib_view']==view and
        [c['name'] for c in metadata['schema']]==spec['ordered_features'], 'column Feature schema/config mismatch')
    source_spec=source_manifest['spec']; source_ref=source_manifest['source_ref']
    validate_source_projection(spec,source_spec)
    require(source_manifest['qlib_view']==view and source_spec['universe']==spec['universe'] and
            source_spec['snapshot']==spec['snapshot'] and source_spec['pit_policy']==spec['pit_policy'] and
            metadata['source_descriptor']['source_ref']==source_ref and view['universe_name']==universe_id,
            'column Feature physical source scope mismatch')
    require(type(rows) is list and all(type(r) is dict and set(r)==ROW_FIELDS for r in rows),
            'exact column Feature rows required')
    indexed=grid(rows,days,spec['universe'],'session')
    require([(r['security_id'],r['session']) for r in rows]==[(s,d) for d in days for s in spec['universe']] and
        metadata['rows']==[{k:v for k,v in r.items() if k!='values'} for r in rows], 'column Feature row order mismatch')
    require(type(metadata['input_evidence']) is list and len(metadata['input_evidence'])==len(days) and
        metadata['day_outputs']==outputs,'column Feature proof/output count mismatch')
    selections=[]; reachable=set(); parents={}; qlib_ref=digest(view)
    for day,proof,output in zip(days,metadata['input_evidence'],outputs):
        require(output['session']==day,'column Feature output date mismatch')
        _fields(proof,{'selection','selection_ref','core_header','day_evidence_ref'},'exact column Feature proof required')
        _verify_ref(proof,'day_evidence_ref'); selection=proof['selection']; header=proof['core_header']
        _fields(selection,SELECTION_FIELDS,'exact daily column source selection required')
        history=ordered(selection['history_sessions'],'column Feature history')
        target_end=spec['read_sessions'].index(day)+1
        require(target_end>=len(history) and spec['read_sessions'][target_end-len(history):target_end]==history,
                'column Feature history outside declared target read sessions')
        require(selection['source_ref']==source_ref and selection['session']==day and
            selection['anchor_session']==day and history[-1]==day and day in source_spec['feature_sessions'] and
            proof['selection_ref']==digest(selection),'column Feature selection identity/anchor mismatch')
        end=source_spec['read_sessions'].index(day)+1
        require(source_spec['read_sessions'][max(0,end-len(history)):end]==history and
            _instant(selection['knowledge_cutoff'])==_instant(spec['cutoff_by_session'][day])==
            _instant(source_spec['cutoff_by_session'][day]), 'column Feature history/cutoff mismatch')
        require(type(header) is dict and all(_ref(header.get(k)) for k in
            ('result_ref','plan_ref','compiled_ref','implementation_ref')),
            'actual packed Core result/plan/implementation refs required')
        refs=header.get('input_refs')
        require(refs=={'source_ref':source_ref,'selection_ref':proof['selection_ref']},
            'packed Core input selection refs mismatch')
        # Preserve the owner's header names. The caller passes the real output
        # header; no legacy FactBatch or FeatureFrame identity is invented.
        core_schema=header.get('schema',header.get('output_schema'))
        require(core_schema==metadata['schema'] and type(header.get('source_bindings')) is list and
            bool(header['source_bindings']) and all(type(b) is dict and b.get('view_ref')==source_ref
                for b in header['source_bindings']),
            'packed Core schema/source scope mismatch')
        for security in spec['universe']:
            row=indexed[security,day]
            require(type(row['member']) is bool and all(type(row[k]) is list and len(row[k])==width
                for k in ('values','validity','availability','reasons')) and
                all(type(v) is bool for v in row['validity']) and
                all(type(r) is list and all(type(x) is str for x in r) for r in row['reasons']), 'column Feature row flags mismatch')
            require(all(v is None or _finite(v) for v in row['values']) and
                all((v is not None)==valid for v,valid in zip(row['values'],row['validity'])), 'column Feature null/validity mismatch')
            require(_instant(row['knowledge_cutoff'])==_instant(selection['knowledge_cutoff']) and
                all(a is None or _instant(a)<=_instant(row['knowledge_cutoff']) for a in row['availability']),
                'column Feature clock conflict')
            require(row['source_refs']==[header['result_ref'],header['plan_ref']], 'column Feature packed Core source mismatch')
        _validate_core_header(header,[indexed[s,day] for s in spec['universe']],selection,spec,metadata['schema'])
        feature_ref=_day_ref(proof,output,spec,view,metadata['schema'])
        parents[day]={'feature_ref':feature_ref,'qlib_view_ref':qlib_ref}
        selections.append({**selection,'selection_ref':proof['selection_ref'],
            'core_result_ref':header['result_ref'],'feature_ref':feature_ref})
        reachable.update((source_ref,proof['selection_ref'],proof['day_evidence_ref'],feature_ref,
                          output['output_ref'],header['result_ref'],header['plan_ref'],header['compiled_ref'],header['implementation_ref']))
    require(metadata['row_references']==parents,'column Feature daily identity mismatch')
    return selections,reachable,parents


def write_block(target, *, rows, proof, source_descriptor, spec, view, schema, index_ref,
                row_offset, options, universe_id, caller_retained_bytes=None, metrics=None,
                source_store=None):
    """Pack only this output block; the shared physical source stays external."""
    from .stock_matrix_reader import _resident_size
    external=0 if caller_retained_bytes is None else caller_retained_bytes()
    require(type(external) is int and external>=0,'nonnegative column writer retained bytes required')
    owned=_resident_size([rows,proof,spec,view,schema])
    require(external+6*owned+len(spec['universe'])*len(schema)*18+8192<=options['maximum_resident_bytes'],
            'column Feature writer snapshot budget exceeded')
    manifest,_=verify_feature_column_source(source_descriptor,store=source_store)
    rows,proof=deepcopy([rows,proof]); days=[p['selection']['session'] for p in proof]
    for p in proof:
        _fields(p,{'selection','selection_ref','core_header'},'exact unsealed packed Core proof required')
    by_key={(r['security_id'],r['session']):r for r in rows}
    require(len(by_key)==len(rows),'duplicate column Feature row')
    rows=[by_key[s,d] for d in days for s in spec['universe']]
    evidence=[seal(p,'day_evidence_ref') for p in proof]
    outputs=[]; parents={}; width=len(schema); count=len(spec['universe'])
    for pos,p in enumerate(evidence):
        day_rows=rows[pos*count:(pos+1)*count]; packed=_row_buffers(day_rows,width)
        hashes={name:'sha256:'+sha256(memoryview(value).cast('B')).hexdigest() for name,value in packed.items()}
        output=_output(p['selection']['session'],day_rows,schema,hashes); outputs.append(output)
        parents[p['selection']['session']]={'feature_ref':_day_ref(p,output,spec,view,schema),'qlib_view_ref':digest(view)}
        del packed
    metadata=seal({'contract_version':METADATA_VERSION,'sessions':days,'ordered_features':spec['ordered_features'],
        'catalog_ref':spec['catalog_ref'],'selection':spec['feature_selection'],'qlib_view':view,'schema':schema,
        'rows':[{k:v for k,v in r.items() if k!='values'} for r in rows],'row_references':parents,
        'source_descriptor':deepcopy(source_descriptor),'input_evidence':evidence,'day_outputs':outputs},'metadata_ref')
    selections,_,_=validate_metadata(metadata,rows,spec,view,manifest,outputs,universe_id=universe_id)
    md=write_part(target,metadata,'metadata_ref'); parts=[]
    require(Path(md['path']).stat().st_size<=64*1024**2,'column Feature metadata parent budget exceeded')
    for start in range(0,width,options['column_block']):
        stop=min(start+options['column_block'],width); shape=[len(rows),stop-start]
        buffers={name:write_buffer(target,(r['values'][j] if r['values'][j] is not None else 0.0
            for r in rows for j in range(start,stop)),dtype='float64_le',shape=shape) for name in ['values']}
        for name,field,dtype in [('value_validity','validity','bool_u8'),
            ('available_at_utc_us','availability','int64_le'),('available_at_validity','availability','bool_u8')]:
            values=(instant_us(r[field][j]) if r[field][j] is not None else 0
                    for r in rows for j in range(start,stop)) if name=='available_at_utc_us' else (
                r[field][j] is not None if name=='available_at_validity' else r[field][j]
                for r in rows for j in range(start,stop))
            buffers[name]=write_buffer(target,values,dtype=dtype,shape=shape)
        parts.append(write_partition(target,table='features',fold_spec_ref=None,row_index_ref=index_ref,
            schema_digest=digest(schema),row_offset=row_offset,row_count=len(rows),
            columns=spec['ordered_features'][start:stop],buffers=buffers,metadata=md))
    return parts,selections


def admit_metadata(store, metadata, rows, parts, spec, view, *, universe_id):
    manifest,reachable=verify_feature_column_source(metadata['source_descriptor'],store=store)
    access={column:(arrays,j) for part,arrays in parts for j,column in enumerate(part['columns'])}
    require(set(access)==set(spec['ordered_features']),'column Feature partition coverage mismatch')
    count=len(spec['universe']); outputs=[]
    for pos,day in enumerate(metadata['sessions']):
        begin=pos*count; hashes={name:sha256() for name in BUFFER_TYPES}
        for i in range(begin,begin+count):
            for column in spec['ordered_features']:
                arrays,j=access[column]
                for name,h in hashes.items(): h.update(arrays[name][i:i+1,j:j+1].tobytes(order='C'))
        outputs.append(_output(day,rows[begin:begin+count],metadata['schema'],
            {name:'sha256:'+h.hexdigest() for name,h in hashes.items()}))
    selections,hashes,parents=validate_metadata(metadata,rows,spec,view,manifest,outputs,universe_id=universe_id)
    return selections,reachable|hashes,parents

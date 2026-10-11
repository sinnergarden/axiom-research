"""Compact Feature matrix source representation; no Data/Core execution.

The existing matrix Store owns file admission. This module only packs the new
Feature children and checks their daily source/output bindings.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import math
import tempfile

from .stock_artifacts import digest, _verify_ref
from .stock_fold_inputs import require, seal, ordered, grid, file_fingerprint
from .stock_label_contracts import _instant
from .stock_matrix_storage import _buffer_array, instant_us, write_buffer, write_part, write_partition

METADATA_VERSION = 'stock_matrix_feature_metadata_v2'
METADATA_FIELDS = {'contract_version','sessions','ordered_features','catalog_ref','selection',
    'qlib_view','schema','rows','row_references','input_evidence','query_contexts','day_outputs','metadata_ref'}
EVIDENCE_FIELDS = {'contract_version','session','core_frame_ref','core_plan','fact_ref','context_ref',
    'sessions','cutoffs','adjusted_input_ref','membership_ref','source_evidence','day_evidence_ref'}
BUFFER_TYPES = {'values':'float64_le','value_validity':'bool_u8',
    'available_at_utc_us':'int64_le','available_at_validity':'bool_u8'}
SELECTION_FIELDS = {'session','cutoff','history_sessions','adjustment_anchor','query_refs',
                    'day_evidence_ref','feature_ref'}


def _fields(value, fields, message):
    require(type(value) is dict and set(value) == set(fields), message)


def _ref(value):
    return (type(value) is str and len(value)==71 and value.startswith('sha256:') and
            all(c in '0123456789abcdef' for c in value[7:]))


def _same_json(left, right):
    """Exact typed comparison of private coverage, including binary64 -0.0."""
    if type(left) is not type(right): return False
    if left is right: return True  # Aliases of the builder's private snapshot.
    if type(left) is dict:
        return left.keys()==right.keys() and all(_same_json(v,right[k]) for k,v in left.items())
    if type(left) in (list,tuple):
        return len(left)==len(right) and all(_same_json(a,b) for a,b in zip(left,right))
    if type(left) is float: return left.hex()==right.hex()
    return left==right


class _FeatureCoverageWriter:
    """Bounded, builder-owned coverage encoding reuse, never a load receipt."""
    def __init__(self):
        self.entries=[]; self.resident_bytes=0; self.published={}

    def write(self, root, value, *, maximum, retained, metrics):
        from .stock_matrix_reader import _resident_size
        from .stock_native_json import _Budget, _json_chunks, _publish, _file_hash
        for original,desc,mark,_ in self.entries:
            if _same_json(original,value):
                require(file_fingerprint(desc['path'])==mark,'saved Feature coverage changed')
                metrics['feature_coverage_reuses']=metrics.get('feature_coverage_reuses',0)+1
                return deepcopy(desc)
        budget=_Budget(maximum,retained,metrics)
        buffers=Path(root)/'buffers'; buffers.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(prefix='.coverage-',dir=buffers,delete=False) as stream:
                temporary=Path(stream.name); h=sha256(); size=0
                metrics['feature_coverage_encode_calls']=metrics.get('feature_coverage_encode_calls',0)+1
                for chunk in _json_chunks(value,budget=budget,registry={},path_resolver=None):
                    size+=len(chunk)
                    require(size<=8*1024**3,'Feature coverage source byte budget exceeded')
                    h.update(chunk); stream.write(chunk)
            ref='sha256:'+h.hexdigest(); path=buffers/(ref[7:]+'.bin')
            self.published[ref]=size
            require(sum(self.published.values())<=8*1024**3,'Feature coverage source byte budget exceeded')
            created=_publish(temporary,path)
            temporary.unlink(); temporary=None
            require(not path.is_symlink() and path.stat().st_size==size,'published Feature coverage size/path mismatch')
            if not created:
                # Existing/orphan/concurrent bytes need independent admission.
                # New private bytes get their hash+grammar together in Store.
                require(_file_hash(path,budget)==ref,'published Feature coverage mismatch')
                metrics['feature_coverage_existing_hash_calls']=metrics.get('feature_coverage_existing_hash_calls',0)+1
            desc={'path':str(path.resolve()),'file_digest':ref,'dtype':'uint8',
                  'shape':[size],'buffer_digest':ref}
            metrics['feature_coverage_encode_bytes']=metrics.get('feature_coverage_encode_bytes',0)+size
            # This graph is the writer's private deepcopy, never caller-owned.
            charge=_resident_size([value,desc])+512
            cap=min(maximum//8,64*1024**2)
            while self.entries and self.resident_bytes+charge>cap:
                self.resident_bytes-=self.entries.pop(0)[3]
            if charge<=cap:
                require(retained()+charge<=maximum,'Feature coverage reuse resident budget exceeded')
                self.entries.append((value,desc,file_fingerprint(path),charge)); self.resident_bytes+=charge
            return deepcopy(desc)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)


def _row_buffers(rows, width):
    return {
        'values':_buffer_array((r['values'][j] if r['values'][j] is not None else 0.0
                               for r in rows for j in range(width)),'float64_le'),
        'value_validity':_buffer_array((v for r in rows for v in r['validity']),'bool_u8'),
        'available_at_utc_us':_buffer_array((instant_us(a) if a is not None else 0
                                           for r in rows for a in r['availability']),'int64_le'),
        'available_at_validity':_buffer_array((a is not None for r in rows for a in r['availability']),'bool_u8')}


def _output(day, rows, schema, hashes):
    return seal({'session':day,'row_count':len(rows),'schema_digest':digest(schema),
        'rows_ref':digest([{k:v for k,v in r.items() if k!='values'} for r in rows]),
        'buffers':{name:{'dtype':dtype,'shape':[len(rows),len(schema)],'buffer_digest':hashes[name]}
                   for name,dtype in BUFFER_TYPES.items()}},'output_ref')


def _day_ref(evidence, output, spec, view, *, _qlib_ref=None):
    return digest({'contract_version':'stock_feature_day_v2','session':evidence['session'],
        'catalog_ref':spec['catalog_ref'],'selection':spec['feature_selection'],
        'ordered_features':spec['ordered_features'],'qlib_view_ref':digest(view) if _qlib_ref is None else _qlib_ref,
        'day_evidence_ref':evidence['day_evidence_ref'],'output_ref':output['output_ref']})


def _selection(evidence, feature_ref, contexts):
    return {'session':evidence['session'],'cutoff':evidence['cutoffs'][evidence['session']],
        'history_sessions':evidence['sessions'],'adjustment_anchor':evidence['session'],
        'query_refs':sorted({digest(contexts[ref]['context']['query'])
                             for ref in {s['query_context_ref'] for s in evidence['source_evidence'].values()}}),
        'day_evidence_ref':evidence['day_evidence_ref'],'feature_ref':feature_ref}


def validate_metadata(metadata, rows, spec, view, contexts, outputs, *, universe_id):
    from .stock_feature_inputs import FEATURE_ROW_FIELDS, _validate_feature_block
    _fields(metadata,METADATA_FIELDS,'exact compact Feature metadata required')
    require(metadata['contract_version']==METADATA_VERSION,'compact Feature metadata version mismatch')
    _verify_ref(metadata,'metadata_ref')
    days=ordered(metadata['sessions'],'compact Feature sessions'); width=len(spec['ordered_features'])
    require(metadata['ordered_features']==spec['ordered_features'] and metadata['catalog_ref']==spec['catalog_ref'] and
            metadata['selection']==spec['feature_selection'] and metadata['qlib_view']==view and
            [c['name'] for c in metadata['schema']]==spec['ordered_features'],'compact Feature schema/source mismatch')
    require(type(rows) is list and all(type(r) is dict and set(r)==FEATURE_ROW_FIELDS for r in rows),
            'exact compact Feature rows required')
    indexed=grid(rows,days,spec['universe'],'session')
    require([(r['security_id'],r['session']) for r in rows]==[(s,d) for d in days for s in spec['universe']] and
            metadata['rows']==[{k:v for k,v in r.items() if k!='values'} for r in rows],
            'compact Feature row order/metadata mismatch')
    evidence=metadata['input_evidence']
    require(type(evidence) is list,'compact Feature proof list required')
    for p in evidence: _fields(p,EVIDENCE_FIELDS,'exact compact Feature day evidence required')
    require([p['session'] for p in evidence]==days,'compact Feature proof date mismatch')
    used=set(); by_day={}; parents={}; selections=[]; reachable=set(); qlib_ref=digest(view)
    require(_same_json(metadata['day_outputs'],outputs),'compact Feature daily output bytes mismatch')
    for p,out in zip(evidence,outputs):
        _fields(p,EVIDENCE_FIELDS,'exact compact Feature day evidence required')
        require(p['contract_version']=='stock_feature_day_evidence_v2','compact Feature evidence version mismatch')
        _verify_ref(p,'day_evidence_ref')
        require(all(_ref(p[k]) for k in ('core_frame_ref','fact_ref','context_ref','adjusted_input_ref','membership_ref')),
                'compact Feature original input refs required')
        require(type(p['source_evidence']) is dict and bool(p['source_evidence']),'compact Feature source bindings required')
        for source in p['source_evidence'].values():
            _fields(source,{'field','batch_ref','query_context_ref','provenance_by_key_ref'},'exact compact Feature source binding required')
            require(source['query_context_ref'] in contexts and _ref(source['batch_ref']) and
                    _ref(source['provenance_by_key_ref']),'compact Feature source evidence closure mismatch')
            used.add(source['query_context_ref'])
        require([c for c in (o['column'] for o in p['core_plan']['outputs'])]==metadata['schema'],
                'compact Feature schema differs from Core output')
        plan_ref=digest(p['core_plan']); day=p['session']; by_day[day]=p
        for s in spec['universe']:
            r=indexed[s,day]
            require(type(r['member']) is bool and all(type(r[k]) is list and len(r[k])==width
                    for k in ('values','validity','availability','reasons')) and
                    all(type(v) is bool for v in r['validity']) and
                    all(type(v) is list and all(type(x) is str for x in v) for v in r['reasons']),
                    'compact Feature row flags/schema mismatch')
            require(all(v is None or type(v) in (int,float) and math.isfinite(v) for v in r['values']) and
                    all((v is not None)==valid for v,valid in zip(r['values'],r['validity'])),
                    'compact Feature logical null/value mismatch')
            require(_instant(r['knowledge_cutoff'])==_instant(spec['cutoff_by_session'][day]) and
                    all(a is None or _instant(a)<=_instant(r['knowledge_cutoff']) for a in r['availability']),
                    'compact Feature clock conflict')
            require(r['source_refs']==[p['core_frame_ref'],plan_ref],'compact Feature Core source mismatch')
        ref=_day_ref(p,out,spec,view,_qlib_ref=qlib_ref)
        parents[day]={'feature_ref':ref,'qlib_view_ref':qlib_ref}
        selection=_selection(p,ref,contexts); selections.append(selection)
        reachable.update(selection['query_refs']); reachable.update((ref,p['day_evidence_ref']))
    require(used==set(contexts),'unused/missing compact Feature context')
    require(metadata['row_references']==parents,'compact Feature day identity mismatch')
    # The original per-day PIT, query, source qualification and anchor validator
    # reads the admitted small context body, without restoring legacy coverage.
    _validate_feature_block({'qlib_view':view},indexed,by_day,spec,view,universe_id,source_contexts=contexts)
    return selections,reachable,parents


def write_block(target, *, rows, proof, spec, view, schema, index_ref, row_offset, options,
                universe_id, caller_retained_bytes=None, metrics=None, coverage_writer=None):
    from .stock_matrix_reader import _resident_size
    metrics={} if metrics is None else metrics
    coverage_writer=coverage_writer or _FeatureCoverageWriter()
    maximum=options['maximum_resident_bytes']; incoming=[rows,proof,spec,view,schema]
    owned=_resident_size(incoming)
    def retained():
        external=0 if caller_retained_bytes is None else caller_retained_bytes()
        require(type(external) is int and external>=0,'nonnegative compact writer retained bytes required')
        return external+4*owned+coverage_writer.resident_bytes
    require(retained()+2*owned+8192<=maximum,'compact Feature snapshot budget exceeded')
    rows,proof,spec,view,schema=deepcopy(incoming)
    days=[p['session'] for p in proof]; by_key={(r['security_id'],r['session']):r for r in rows}
    require(len(by_key)==len(rows),'duplicate compact Feature row')
    rows=[by_key[s,d] for d in days for s in spec['universe']]
    context_values={}; context_descs={}; local=[]; evidence=[]
    for p in proof:
        sources={}
        for source_id,source in p['source_evidence'].items():
            _fields(source,{'field','batch_ref','query_context','provenance_by_key_ref'},'original compact writer source proof required')
            ctx=source['query_context']; found=next((desc for original,desc in local if _same_json(original,ctx)),None)
            if found is None:
                coverage={'present':'coverage' in ctx,'bytes':None}
                if coverage['present']:
                    coverage['bytes']=coverage_writer.write(target,ctx['coverage'],maximum=maximum,retained=retained,metrics=metrics)
                child=seal({'contract_version':'stock_feature_query_context_v1',
                    'context':{k:v for k,v in ctx.items() if k!='coverage'},'coverage':coverage},'query_context_ref')
                found=write_part(target,child,'query_context_ref')
                require(Path(found['path']).stat().st_size<=64*1024**2,'compact Feature context parent budget exceeded')
                context_values[child['query_context_ref']]=child; context_descs[child['query_context_ref']]=found
                local.append((ctx,found))
                metrics['feature_context_write_calls']=metrics.get('feature_context_write_calls',0)+1
            sources[source_id]={k:v for k,v in source.items() if k!='query_context'}
            sources[source_id]['query_context_ref']=found['query_context_ref']
        day={k:v for k,v in p.items() if k!='source_evidence'}
        day.update(contract_version='stock_feature_day_evidence_v2',source_evidence=sources)
        evidence.append(seal(day,'day_evidence_ref'))
    outputs=[]; parents={}; width=len(schema); universe=len(spec['universe']); qlib_ref=digest(view)
    require(retained()+universe*width*18+8192<=maximum,'compact Feature daily buffer budget exceeded')
    for pos,p in enumerate(evidence):
        day_rows=rows[pos*universe:(pos+1)*universe]; packed=_row_buffers(day_rows,width)
        hashes={name:'sha256:'+sha256(memoryview(value).cast('B')).hexdigest() for name,value in packed.items()}
        out=_output(p['session'],day_rows,schema,hashes); outputs.append(out)
        parents[p['session']]={'feature_ref':_day_ref(p,out,spec,view,_qlib_ref=qlib_ref),'qlib_view_ref':qlib_ref}
        del packed
    metadata=seal({'contract_version':METADATA_VERSION,'sessions':days,'ordered_features':spec['ordered_features'],
        'catalog_ref':spec['catalog_ref'],'selection':spec['feature_selection'],'qlib_view':view,'schema':schema,
        'rows':[{k:v for k,v in r.items() if k!='values'} for r in rows],
        'row_references':parents,'input_evidence':evidence,
        'query_contexts':[context_descs[k] for k in sorted(context_descs)],'day_outputs':outputs},'metadata_ref')
    selections,_,_=validate_metadata(metadata,rows,spec,view,context_values,outputs,universe_id=universe_id)
    md=write_part(target,metadata,'metadata_ref')
    require(Path(md['path']).stat().st_size<=64*1024**2,'compact Feature metadata parent budget exceeded')
    parts=[]
    for start in range(0,width,options['column_block']):
        stop=min(start+options['column_block'],width); shape=[len(rows),stop-start]
        buffers={
            'values':write_buffer(target,(r['values'][j] if r['values'][j] is not None else 0.0
                for r in rows for j in range(start,stop)),dtype='float64_le',shape=shape),
            'value_validity':write_buffer(target,(r['validity'][j] for r in rows for j in range(start,stop)),
                dtype='bool_u8',shape=shape),
            'available_at_utc_us':write_buffer(target,(instant_us(r['availability'][j]) if r['availability'][j] is not None else 0
                for r in rows for j in range(start,stop)),dtype='int64_le',shape=shape),
            'available_at_validity':write_buffer(target,(r['availability'][j] is not None
                for r in rows for j in range(start,stop)),dtype='bool_u8',shape=shape)}
        parts.append(write_partition(target,table='features',fold_spec_ref=None,row_index_ref=index_ref,
            schema_digest=digest(schema),row_offset=row_offset,row_count=len(rows),
            columns=spec['ordered_features'][start:stop],buffers=buffers,metadata=md))
    return parts,selections


def admit_metadata(store, metadata, rows, parts, spec, view, *, universe_id):
    _fields(metadata,METADATA_FIELDS,'exact compact Feature metadata required')
    descriptors=metadata['query_contexts']
    require(type(descriptors) is list and bool(descriptors),'compact Feature contexts required')
    refs=[d['query_context_ref'] for d in descriptors]
    require(refs==sorted(set(refs)),'unique ordered compact Feature contexts required')
    contexts={}
    for desc in descriptors:
        child=store.read_json(desc,'query_context_ref',_revisit=True)
        _fields(child,{'contract_version','context','coverage','query_context_ref'},'exact Feature query context required')
        require(child['contract_version']=='stock_feature_query_context_v1' and type(child['context']) is dict and
                'coverage' not in child['context'],'compact Feature context version/body mismatch')
        _fields(child['coverage'],{'present','bytes'},'exact compact Feature coverage required')
        coverage=child['coverage']; require(type(coverage['present']) is bool,'compact coverage presence must be bool')
        if coverage['present']: store._admit_coverage(coverage['bytes'],parent=child)
        else: require(coverage['bytes'] is None,'absent compact coverage must be null')
        contexts[desc['query_context_ref']]=child
    # Feed each row's columns in complete F order. Concatenating whole column
    # blocks would incorrectly interleave rows when U>1 and there are 2 blocks.
    columns=spec['ordered_features']; access={}
    for p,arrays in parts:
        for j,column in enumerate(p['columns']): access[column]=(arrays,j)
    runs=[]
    for column in columns:
        arrays,j=access[column]
        if runs and runs[-1][0] is arrays and runs[-1][2]==j: runs[-1][2]=j+1
        else: runs.append([arrays,j,j+1])
    universe=len(spec['universe']); outputs=[]
    for pos,day in enumerate(metadata['sessions']):
        begin=pos*universe; hashes={name:sha256() for name in BUFFER_TYPES}
        for i in range(begin,begin+universe):
            for arrays,start,stop in runs:
                for name,h in hashes.items(): h.update(arrays[name][i:i+1,start:stop].tobytes(order='C'))
        outputs.append(_output(day,rows[begin:begin+universe],metadata['schema'],
            {name:'sha256:'+h.hexdigest() for name,h in hashes.items()}))
    result=validate_metadata(metadata,rows,spec,view,contexts,outputs,universe_id=universe_id)
    for desc in descriptors: store.drop_json(desc,'query_context_ref')
    return result

"""Bounded ordinary v1 Feature parents; saved loading uses only the stdlib."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import errno
import math
import tempfile

from .stock_artifacts import digest, file_digest, _read, _verify_ref, write_json
from .stock_fold_inputs import (require, ordered, read_parent, seal,
                               validate_feature_parent, file_fingerprint)
from .stock_label_contracts import _instant, _session

DEFAULT_LIMITS = {'maximum_parent_bytes': 64 * 1024**2}
SPEC_FIELDS = {'contract_version','scope','snapshot','pit_policy','calendar','universe',
    'catalog_ref','feature_selection','ordered_features','read_sessions','feature_sessions','cutoff_by_session'}
INDEX_FIELDS = {'contract_version','definition','definition_ref','feature_inputs_ref','content_digest',
    'status','qlib_view','qlib_manifest','feature_parents'}
QLIB_KEYS = ('schema_version','view_id','snapshot_id','queries','fields','instrument_map','calendar',
             'universe_query','universe_name','numeric_format','reader_version','exporter_version','limitations')


def _limits(limits):
    value = deepcopy(DEFAULT_LIMITS if limits is None else limits)
    require(type(value) is dict and set(value) == set(DEFAULT_LIMITS) and
            all(type(v) is int and v > 0 for v in value.values()), 'positive parent byte limit required')
    return value


def _bounded(path, limits):
    require(Path(path).stat().st_size <= limits['maximum_parent_bytes'], 'parent byte limit exceeded')


def _spec(spec, *, reader=read_parent):
    require(type(spec) is dict and set(spec) == SPEC_FIELDS and
            spec['contract_version'] == 'stock_feature_inputs_spec_v1', 'exact Feature input spec required')
    require(type(spec['snapshot']) is str and spec['snapshot'] not in ('','current','latest') and
            type(spec['pit_policy']) is str and bool(spec['pit_policy']), 'fixed Snapshot/PIT required')
    calendar = ordered(spec['calendar'], 'calendar'); [_session(d) for d in calendar]
    ordered(spec['universe'], 'universe'); read = ordered(spec['read_sessions'], 'read sessions')
    days = ordered(spec['feature_sessions'], 'Feature dates')
    require(set(read) <= set(calendar) and set(days) <= set(read), 'Feature dates outside frozen calendar')
    require(read == calendar[calendar.index(read[0]):calendar.index(read[-1])+1] and
            read[-1] == days[-1], 'complete actual read range required')
    require(set(spec['cutoff_by_session']) == set(read) and all(
        _instant(spec['cutoff_by_session'][d]) == _instant(d+'T20:30:00+08:00') for d in read),
        'original per-session Feature clocks required')
    columns = spec['ordered_features']; selection = spec['feature_selection']
    require(type(spec['catalog_ref']) is str and spec['catalog_ref'].startswith('sha256:') and
            len(spec['catalog_ref']) == 71, 'fixed Feature catalog ref required')
    require(type(columns) is list and bool(columns) and all(type(c) is str and bool(c) for c in columns) and
            len(set(columns)) == len(columns) and
            type(selection) is list and len(selection) == len(columns) and
            all(type(v) is dict and set(v) == {'id','semantic_version'} and
                type(v['semantic_version']) is str and bool(v['semantic_version']) for v in selection) and
            [v['id'] for v in selection] == columns, 'ordered Feature selection required')
    scope = reader(spec['scope'], 'scope_bundle_ref')
    for key, name in (('request_ref','request'),('result_ref','result'),('source_proof_ref','source_proof')):
        require(scope[key] == digest(scope[name]), 'scope source linkage mismatch')
    result = scope['result']; request = scope['request']
    require(result['read_sessions'] == calendar and result['read_symbols'] == spec['universe'] and
            result['snapshot_id'] == spec['snapshot'] and result['pit_policy'] == spec['pit_policy'],
            'Feature scope projection mismatch')
    universe_id = request.get('universe_id')
    require(type(universe_id) is str and bool(universe_id) and request.get('snapshot') == spec['snapshot'] and
            request.get('pit_policy') == spec['pit_policy'], 'original scope universe/Snapshot/PIT required')
    return universe_id


def _qlib(descriptor, limits, *, fingerprints=None, _file_hasher=None, _json_reader=None):
    """Verify the saved Data manifest and its actual files, without importing Data."""
    _file_hasher=file_digest if _file_hasher is None else _file_hasher
    _json_reader=_read if _json_reader is None else _json_reader
    require(type(descriptor) is dict and set(descriptor) == {'path','file_digest','view_id'} and
            Path(descriptor['path']).is_absolute(), 'explicit saved Qlib manifest required')
    path = Path(descriptor['path']); _bounded(path, limits)
    marks = {str(path):file_fingerprint(path)}
    require(_file_hasher(path) == descriptor['file_digest'], 'Qlib manifest file mismatch')
    manifest = _json_reader(path)
    require(manifest['schema_version'] == 'axiom_qlib_view_v1' and
            manifest['view_id'] == descriptor['view_id'] == digest(
                {k:v for k,v in manifest.items() if k != 'view_id'})[7:], 'Qlib identity mismatch')
    root = path.parent.resolve()
    marks[str(root)] = file_fingerprint(root)
    marks.update({str(p):file_fingerprint(p) for p in root.rglob('*') if p.is_dir()})
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} ==
            set(manifest['files']) | {path.name}, 'Qlib file coverage mismatch')
    for relative, expected in manifest['files'].items():
        child = root / relative
        marks[str(child)] = file_fingerprint(child)
        require(not child.is_symlink() and root in child.resolve().parents and
                child.stat().st_size == expected['size'] and
                _file_hasher(child) == 'sha256:'+expected['sha256'], 'Qlib file mismatch: '+relative)
    require(all(file_fingerprint(p) == mark for p,mark in marks.items()),
            'Qlib files changed during read')
    if fingerprints is not None: fingerprints.update(marks)
    return {key:manifest[key] for key in QLIB_KEYS}


def _qlib_scope(view, spec, universe_id):
    require(view['snapshot_id'] == spec['snapshot'] and view['calendar'] == spec['read_sessions'] and
            set(view['instrument_map']) == set(spec['universe']) and view['universe_name'] == universe_id,
            'Qlib scope mismatch')
    queries = view['queries'] + [view['universe_query']]
    require(len(queries) == 3 and all(q['symbols'] == spec['universe'] and
            q['sessions'] == spec['read_sessions'] and q['pit_policy'] == spec['pit_policy'] and
            q['purpose'] == 'decision_facts' and q['price_basis'] == 'unadjusted' and
            q.get('policy_by_session') is None and q.get('adjustment_anchor') is None and
            set(q['cutoff_by_session']) == set(spec['read_sessions']) and all(
                _instant(q['cutoff_by_session'][d]) == _instant(spec['cutoff_by_session'][d])
                for d in spec['read_sessions']) for q in queries), 'Qlib original query mismatch')
    require([(q['domain'],q['fields']) for q in queries] == [
        ('market_daily',['open','high','low','close','amount_cny']),
        ('adjustment_factors',['factor']),('universe_membership',['is_member'])] and
        queries[-1]['universe_id'] == universe_id and
        all(q.get('universe_id') is None for q in queries[:-1]), 'Qlib original fields/universe mismatch')


def _validate_feature_block(feature, indexed, proof, spec, view, universe_id):
    """Shared native Feature source/PIT checks for v1 files and v2 blocks."""
    history_length = spec['read_sessions'].index(spec['feature_sessions'][0]) + 1
    require(feature['qlib_view'] == view, 'parent Qlib identity mismatch')
    for day in proof:
        p = proof[day]; pos = spec['read_sessions'].index(day)
        history = spec['read_sessions'][pos-history_length+1:pos+1]
        require(p['sessions'] == history and set(p['cutoffs']) == set(history) and
                all(_instant(c) == _instant(spec['cutoff_by_session'][day]) for c in p['cutoffs'].values()),
                'Feature proof original window/clock mismatch')
        members = p['core_plan'].get('reference_members')
        require(type(members) is dict and day in members, 'Feature original member proof required')
        for security in spec['universe']:
            row = indexed[security,day]
            require(row['member'] is (security in members[day]) and len(row['reasons']) == len(spec['ordered_features']) and
                    all(v is None or type(v) in (int,float) and math.isfinite(v) for v in row['values']),
                    'Feature original member/value schema mismatch')
        sources = p['source_evidence']; bindings = p['core_plan'].get('sources')
        require(type(sources) is dict and bool(sources) and type(bindings) is list and
                all(type(b) is dict and type(b.get('id')) is str for b in bindings) and
                len({b['id'] for b in bindings}) == len(bindings) and
                set(sources) == {b['id'] for b in bindings}, 'Feature evidence/Core source IDs mismatch')
        bound = {b['id']:b for b in bindings}
        fields = ['open','high','low','close','amount_cny']
        require([c['name'] for c in p['core_plan'].get('input_schema',[])] == fields and
                {s['field'] for s in sources.values()} == set(fields+['is_member']),
                'Feature original source field coverage mismatch')
        membership = False
        for source_id,source in sources.items():
            context = source['query_context']; query = context['query']
            require(context['snapshot_id'] == spec['snapshot'] and query['sessions'] == history and
                    query['symbols'] == spec['universe'] and query['pit_policy'] == spec['pit_policy'] and
                    query.get('policy_by_session') is None and
                    query['purpose'] == 'decision_facts' and set(query['cutoff_by_session']) == set(history) and
                    all(_instant(c) == _instant(spec['cutoff_by_session'][day]) for c in query['cutoff_by_session'].values()),
                    'Feature original source query mismatch')
            if source['field'] == 'is_member':
                membership = True
                require(context['domain'] == 'universe_membership' and query['fields'] == ['is_member'] and
                        query['price_basis'] == 'unadjusted' and query.get('adjustment_anchor') is None and
                        query['universe_id'] == universe_id and source['batch_ref'] == p['membership_ref'],
                        'Feature original membership ref mismatch')
            else:
                require(context['domain'] == 'market_daily' and query['price_basis'] == 'common_anchor_adjusted_v1' and
                        query['fields'] == ['open','high','low','close','amount_cny'] and
                        source['field'] in query['fields'] and query.get('universe_id') is None and
                        query['adjustment_anchor'] == day and source['batch_ref'] == p['adjusted_input_ref'],
                        'Feature original adjusted ref mismatch')
            binding = bound[source_id]
            require(source_id == digest({'field':source['field'],'batch_ref':source['batch_ref'],
                'qualification':binding['qualification'],'basis':binding['availability_basis']}) and
                binding['revision_policy'] == query['pit_policy'], 'Feature original field/source binding mismatch')
            reference = digest({'snapshot_id':context['snapshot_id'],'query':query,
                                'reader_version':context['reader_version']})
            data_ref = reference if source['field'] == 'is_member' else digest({
                'snapshot_id':context['snapshot_id'],'domain':context['domain']})
            view_ref = reference if source['field'] == 'is_member' else digest({
                'snapshot_id':context['snapshot_id'],'query':query,
                'reader_version':context['reader_version'],'derivation':context.get('derivation')})
            require(binding['data_ref'] == data_ref and binding['view_ref'] == view_ref,
                    'Feature original source context binding mismatch')
        require(membership, 'Feature membership source missing')


def _parents(parents, spec, view, universe_id, limits, *, complete, expected_dates=None):
    """Same v1 validator plus the new exact-scope source closure; one pair resident."""
    covered = []; marks = {}
    expected_dates = spec['feature_sessions'] if expected_dates is None else expected_dates
    for desc in parents:
        for name in ('features','input_evidence'):
            path = desc[name]['path']; _bounded(path, limits); marks[path] = file_fingerprint(path)
        require(desc['sessions'] == expected_dates[len(covered):len(covered)+len(desc['sessions'])],
                'parent date order/coverage mismatch')
        feature, indexed, proof = validate_feature_parent(desc, spec)
        _validate_feature_block(feature,indexed,proof,spec,view,universe_id)
        covered.extend(desc['sessions'])
        del feature, indexed, proof
    require(not complete or covered == expected_dates, 'incomplete Feature input coverage')
    require(all(file_fingerprint(p) == mark for p,mark in marks.items()), 'Feature parent changed during validation')
    return covered, marks


@dataclass(frozen=True)
class StockFeatureInputs:
    path: Path
    reused: bool = False
    _index: dict = field(default=None,repr=False,compare=False)

    def to_dict(self):
        require(self._index is not None, 'Feature inputs require saved validation')
        return deepcopy(self._index)

    @property
    def identity(self):
        return self.to_dict()['feature_inputs_ref']

    @property
    def feature_parents(self):
        return self.to_dict()['feature_parents']


def load_stock_feature_inputs(path, *, limits=None):
    """Verify complete ordinary files/refs/clocks; no provider or stage execution."""
    limits = _limits(limits); path = Path(path); _bounded(path/'index.json', limits)
    marks = {str(path/'index.json'):file_fingerprint(path/'index.json')}
    value = _read(path/'index.json')
    if value.get('contract_version') == 'stock_feature_inputs_v2':
        from .stock_matrix_reader import load_feature_matrix_index
        return load_feature_matrix_index(path, limits=limits, _index=value,
                                         _index_fingerprint=marks[str(path/'index.json')])
    require(set(value) == INDEX_FIELDS and value['contract_version'] == 'stock_feature_inputs_v1' and
            value['status'] == 'COMPLETE', 'complete Feature input index required')
    _verify_ref(value, 'content_digest'); definition = value['definition']; spec = definition['spec']
    require(set(definition) == {'spec','shard_sessions','implementation_sources','implementation_ref','environment'} and
            definition['implementation_ref'] == digest(definition['implementation_sources']) and
            value['definition_ref'] == digest(definition) and type(definition['shard_sessions']) is int and
            definition['shard_sessions'] > 0, 'Feature input definition mismatch')
    marks[spec['scope']['path']] = file_fingerprint(spec['scope']['path'])
    universe_id = _spec(spec); view = _qlib(value['qlib_manifest'], limits,fingerprints=marks)
    require(view == value['qlib_view'], 'saved Qlib reference mismatch'); _qlib_scope(view,spec,universe_id)
    require(value['feature_inputs_ref'] == digest({k:value[k] for k in
            ('definition_ref','qlib_manifest','feature_parents')}), 'Feature input identity mismatch')
    require(type(value['feature_parents']) is list and bool(value['feature_parents']) and all(
        len(d['sessions']) == min(definition['shard_sessions'],len(spec['feature_sessions'])-i*definition['shard_sessions'])
        for i,d in enumerate(value['feature_parents'])), 'Feature shard size mismatch')
    _parents(value['feature_parents'],spec,view,universe_id,limits,complete=True)
    require(all(file_fingerprint(p) == mark for p,mark in marks.items()),
            'Feature index/source changed during validation')
    return StockFeatureInputs(path,_index=value)


def _atomic(path, value):
    with tempfile.NamedTemporaryFile(prefix='.'+path.name,dir=path.parent,delete=False) as stream:
        temp = Path(stream.name)
    try:
        write_json(temp,value); temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


MATRIX_OPTIONS = {'layout':'matrix_v1', 'row_block_sessions':64, 'column_block':32,
                  'maximum_resident_bytes':int(5.5*1024**3)}
MATRIX_METADATA_FIELDS = {'contract_version','sessions','ordered_features','catalog_ref',
    'selection','qlib_view','schema','rows','row_references','input_evidence',
    'original_feature_ref','contents','metadata_ref'}
FEATURE_ROW_FIELDS = {'security_id','session','values','validity','availability','reasons',
                      'member','knowledge_cutoff','source_refs'}


def _storage_options(options):
    require(type(options) is dict and not (set(options)-set(MATRIX_OPTIONS)),
            'exact matrix storage options required')
    value = {**MATRIX_OPTIONS, **options}
    require(value['layout'] == 'matrix_v1' and all(type(value[k]) is int and value[k] > 0
            for k in ('row_block_sessions','column_block','maximum_resident_bytes')),
            'matrix layout and positive integer budgets required')
    return value


def _feature_wire(rows, proof, spec, view):
    return seal({'contract_version':'stock_feature_build_v1','catalog_ref':spec['catalog_ref'],
        'selection':spec['feature_selection'],'ordered_features':spec['ordered_features'],
        'qlib_view':view,'input_evidence_ref':digest(proof),'rows':rows}, 'feature_ref')


def _feature_contents(proof):
    """Keep Query/version claims reachable in the same admitted metadata."""
    contents = {}; feature_rows = []
    for p in proof:
        queries = []
        for source in p['source_evidence'].values():
            query = source['query_context']['query']; ref = digest(query)
            contents[ref] = query
            if ref not in queries: queries.append(ref)
        versions = p['source_evidence']; selected = digest(versions)
        contents[selected] = versions
        feature_rows.append({'session':p['session'], 'cutoff':p['cutoffs'][p['session']],
            'history_sessions':p['sessions'], 'adjustment_anchor':p['session'],
            'query_refs':sorted(queries), 'selected_versions_ref':selected})
    return contents, feature_rows


def _validate_feature_matrix_block(metadata, rows, spec, view, *, universe_id=None):
    """Admit reconstructed original rows; no Core or storage-side mathematics."""
    from .stock_fold_inputs import grid
    require(type(metadata) is dict and set(metadata) == MATRIX_METADATA_FIELDS and
            metadata['contract_version'] == 'stock_matrix_feature_metadata_v1',
            'exact Feature matrix metadata required')
    _verify_ref(metadata, 'metadata_ref')
    days = ordered(metadata['sessions'], 'matrix Feature block sessions')
    columns = spec['ordered_features']; width = len(columns)
    require(metadata['ordered_features'] == columns and metadata['catalog_ref'] == spec['catalog_ref'] and
            metadata['selection'] == spec['feature_selection'] and metadata['qlib_view'] == view and
            [c.get('name') for c in metadata['schema']] == columns and
            all(type(c) is dict and set(c) == {'name','dtype','unit','stage','missing'} and
                c['dtype'] == 'float64' for c in metadata['schema']),
            'Feature matrix schema/source mismatch')
    require(type(rows) is list and all(type(r) is dict and set(r) == FEATURE_ROW_FIELDS for r in rows),
            'exact Feature matrix rows required')
    indexed = grid(rows, days, spec['universe'], 'session')
    require([(r['security_id'],r['session']) for r in rows] ==
            [(security,day) for day in days for security in spec['universe']],
            'Feature matrix row order mismatch')
    require(metadata['rows'] == [{k:v for k,v in r.items() if k != 'values'} for r in rows],
            'Feature matrix metadata/flags mismatch')
    proof = metadata['input_evidence']
    require(type(proof) is list and [p['session'] for p in proof] == days,
            'Feature matrix proof date coverage mismatch')
    by_day = {p['session']:p for p in proof}
    for p in proof:
        if 'outputs' in p['core_plan']:
            require([output['column'] for output in p['core_plan']['outputs']] == metadata['schema'],
                    'Feature matrix schema differs from original Core output')
    feature = _feature_wire(rows, proof, spec, view)
    require(feature['feature_ref'] == metadata['original_feature_ref'], 'Feature matrix original slice mismatch')
    require(metadata['row_references'] == {day:{'feature_ref':_feature_wire(
            [indexed[s,day] for s in spec['universe']], [by_day[day]], spec, view)['feature_ref'],
            'qlib_view_ref':digest(view)} for day in days}, 'Feature matrix day slice ref mismatch')
    contents, unused = _feature_contents(proof)
    require(contents == metadata['contents'], 'Feature matrix reachable source evidence mismatch')
    for row in rows:
        require(type(row['member']) is bool and len(row['values']) == len(row['validity']) ==
                len(row['availability']) == len(row['reasons']) == width and
                all(type(v) is bool for v in row['validity']) and
                all(type(reasons) is list and all(type(reason) is str for reason in reasons)
                    for reasons in row['reasons']), 'Feature matrix row schema mismatch')
        require(all(v is None or type(v) in (int,float) and math.isfinite(v) for v in row['values']) and
                all((v is not None) == valid for v,valid in zip(row['values'],row['validity'])),
                'Feature matrix logical null/value mismatch')
        require(_instant(row['knowledge_cutoff']) == _instant(spec['cutoff_by_session'][row['session']]) and
                all(a is None or _instant(a) <= _instant(row['knowledge_cutoff']) for a in row['availability']),
                'original Feature clock conflict')
        p = by_day[row['session']]
        require(row['source_refs'] == [p['core_frame_ref'],digest(p['core_plan'])],
                'Feature matrix Core source mismatch')
    # The original exact Data source/anchor/window and member admission is shared
    # with v1, rather than a looser new packed-path validator.
    if universe_id is None: universe_id = _spec(spec)
    _validate_feature_block(feature,indexed,by_day,spec,view,universe_id)
    return indexed


def _write_feature_matrix_block(target, *, rows, proof, spec, view, schema, index_ref,
                                row_offset, options, universe_id=None):
    from .stock_matrix_storage import instant_us, write_buffer, write_part, write_partition
    days = [p['session'] for p in proof]
    # Core's rows may use its documented key order. Storage always keys them into
    # the one declared day-major grid, preserving every value/flag/reason.
    by_key = {(r['security_id'],r['session']):r for r in rows}
    require(len(by_key) == len(rows), 'duplicate matrix Feature row')
    rows = [by_key[s,d] for d in days for s in spec['universe']]
    original = _feature_wire(rows,proof,spec,view)
    contents, feature_rows = _feature_contents(proof)
    metadata = seal({'contract_version':'stock_matrix_feature_metadata_v1',
        'sessions':days,'ordered_features':spec['ordered_features'],'catalog_ref':spec['catalog_ref'],
        'selection':spec['feature_selection'],'qlib_view':view,'schema':schema,
        'rows':[{k:v for k,v in r.items() if k != 'values'} for r in rows],
        'row_references':{day:{'feature_ref':_feature_wire([by_key[s,day] for s in spec['universe']],
                [p],spec,view)['feature_ref'],'qlib_view_ref':digest(view)} for day,p in zip(days,proof)},
        'input_evidence':proof,'original_feature_ref':original['feature_ref'],'contents':contents}, 'metadata_ref')
    _validate_feature_matrix_block(metadata,rows,spec,view,universe_id=universe_id)
    md = write_part(target,metadata,'metadata_ref'); _bounded(md['path'],DEFAULT_LIMITS); parts=[]
    for start in range(0,len(schema),options['column_block']):
        stop=min(start+options['column_block'],len(schema)); shape=[len(rows),stop-start]
        buffers = {
            'values':write_buffer(target,(r['values'][j] if r['values'][j] is not None else 0.0
                        for r in rows for j in range(start,stop)),dtype='float64_le',shape=shape),
            'value_validity':write_buffer(target,(r['validity'][j] for r in rows for j in range(start,stop)),
                                         dtype='bool_u8',shape=shape),
            'available_at_utc_us':write_buffer(target,(instant_us(r['availability'][j])
                        if r['availability'][j] is not None else 0 for r in rows for j in range(start,stop)),
                                               dtype='int64_le',shape=shape),
            'available_at_validity':write_buffer(target,(r['availability'][j] is not None
                        for r in rows for j in range(start,stop)),dtype='bool_u8',shape=shape),
        }
        parts.append(write_partition(target,table='features',fold_spec_ref=None,row_index_ref=index_ref,
            schema_digest=digest(schema),row_offset=row_offset,row_count=len(rows),
            columns=spec['ordered_features'][start:stop],buffers=buffers,metadata=md))
    return parts, feature_rows


def _object_upper_bytes(value):
    """Conservative per-block storage bound, including repeated JSON leaves.

    Counting shared children again is intentional: canonical JSON repeats their
    contents. This is a bound for storage-owned objects and encoder copies, not
    a claim about the native executor's or process tree's total RSS.
    """
    import sys
    own = sys.getsizeof(value)
    if type(value) is dict:
        return own+sum(_object_upper_bytes(k)+_object_upper_bytes(v) for k,v in value.items())
    if type(value) in (list,tuple): return own+sum(_object_upper_bytes(v) for v in value)
    if type(value) is str: return max(own,len(value.encode('utf-8'))+2)
    return own


def _build_feature_matrix(data, *, spec, destination, storage_options, progress):
    from .feature_catalog import load_feature_catalog
    from .stock_ml import _implementation, _environment, _prepare_stock_qlib, _iter_stock_feature_days
    from .stock_matrix_storage import row_index, write_part
    options = _storage_options(storage_options); spec = deepcopy(spec)
    source_marks={spec['scope']['path']:file_fingerprint(spec['scope']['path'])}
    universe_id=_spec(spec); catalog=load_feature_catalog(); chosen=catalog.select(spec['feature_selection'])
    require(catalog.identity == spec['catalog_ref'] and [c['id'] for c in chosen] == spec['ordered_features'],
            'current Feature catalog mismatch')
    history_length=max(c['lookback'] for c in chosen); calendar=spec['calendar']
    first=calendar.index(spec['feature_sessions'][0])
    require(first >= history_length-1 and spec['read_sessions'] == calendar[
        first-history_length+1:calendar.index(spec['feature_sessions'][-1])+1], 'actual Feature history range mismatch')
    # All current Feature recipes terminate in the existing Core CS schema.
    schema=[{'name':c['id'],'dtype':'float64','unit':'dimensionless','stage':'cross_sectional',
             'missing':'preserve'} for c in chosen]
    implementations=_implementation(); definition={'spec':spec,'storage_options':options,
        'implementation_sources':implementations,'implementation_ref':digest(implementations),'environment':_environment()}
    require(all(file_fingerprint(p) == mark for p,mark in source_marks.items()),
            'Feature scope changed after admission')
    definition_ref=digest(definition); target=Path(destination)/definition_ref[7:]
    if (target/'index.json').exists():
        saved=load_stock_feature_inputs(target)
        require(saved.to_dict()['definition'] == definition, 'cached matrix Feature definition mismatch')
        saved.reused = True
        return saved
    target.mkdir(parents=True,exist_ok=True)
    config={'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],'symbols':spec['universe'],
        'universe_id':universe_id,'calendar':calendar,'read_sessions':spec['read_sessions'],
        'feature_sessions':spec['feature_sessions'],'cutoff_by_session':spec['cutoff_by_session'],
        'feature_selection':spec['feature_selection'],'scope_ref':spec['scope']['scope_bundle_ref']}
    prepared=_prepare_stock_qlib(data,config=config,destination=target); view=prepared['view_reference']
    _qlib_scope(view,spec,universe_id); qpath=prepared['qlib_path']/'axiom-qlib.json'
    qdesc={'path':str(qpath.resolve()),'file_digest':file_digest(qpath),'view_id':view['view_id']}
    require(_qlib(qdesc,DEFAULT_LIMITS,fingerprints=source_marks) == view, 'prepared Qlib reference mismatch')
    row_wire=row_index(spec['feature_sessions'],spec['universe']); rid=write_part(target,row_wire,'row_index_ref')
    partitions=[]; feature_rows=[]; rows=[]; proof=[]; completed=[]; pending_bytes=0
    checkpoint=target/'checkpoint.json'
    if checkpoint.exists():
        from .stock_matrix_reader import VerifiedMatrixStore, admit_feature_matrix_parts
        previous=_read(checkpoint); _verify_ref(previous,'content_digest')
        require(set(previous) == {'contract_version','definition_ref','qlib_manifest','schema','row_index',
            'partitions','feature_rows','content_digest'} and
            previous['contract_version'] == 'stock_feature_inputs_checkpoint_v2' and
            previous['definition_ref'] == definition_ref and previous['qlib_manifest'] == qdesc and
            previous['schema'] == schema and previous['row_index'] == rid, 'matrix Feature checkpoint mismatch')
        with VerifiedMatrixStore(maximum_parent_bytes=DEFAULT_LIMITS['maximum_parent_bytes']) as store:
            completed, admitted_blocks, unused_sources, unused_parents=admit_feature_matrix_parts(
                store,spec=spec,view=view,schema=schema,row_index=row_wire,
                partitions=previous['partitions'],complete=False,universe_id=universe_id)
            require(previous['feature_rows'] == [row for block in admitted_blocks
                for row in block['_source_selection_rows']], 'matrix checkpoint source-selection mismatch')
            source_marks.update(store.fingerprints)
        partitions=previous['partitions']; feature_rows=previous['feature_rows']
        require([v['session'] for v in feature_rows] == completed, 'matrix checkpoint source-selection mismatch')
    config['feature_sessions']=spec['feature_sessions'][len(completed):]
    stats={'data_read_calls':0,'core_calls':0,'feature_core_calls':0}
    # row_block_sessions is a maximum. Large schemas use the same writer with
    # smaller blocks when metadata/encoder residency demands it, not a separate
    # six-column path. The public parent-byte admission stays meaningful.
    block_budget=min(DEFAULT_LIMITS['maximum_parent_bytes'],options['maximum_resident_bytes']//3)
    def publish_block():
        nonlocal rows, proof, pending_bytes
        new_parts,new_sources=_write_feature_matrix_block(target,rows=rows,proof=proof,spec=spec,view=view,
            schema=schema,index_ref=row_wire['row_index_ref'],row_offset=len(completed)*len(spec['universe']),
            options=options,universe_id=universe_id)
        partitions.extend(new_parts); feature_rows.extend(new_sources); completed.extend(p['session'] for p in proof)
        _atomic(checkpoint,seal({'contract_version':'stock_feature_inputs_checkpoint_v2','definition_ref':definition_ref,
            'qlib_manifest':qdesc,'schema':schema,'row_index':rid,'partitions':partitions,
            'feature_rows':feature_rows},'content_digest'))
        rows=[]; proof=[]; pending_bytes=0
        if progress is not None: progress({'stage':'feature_matrix_block','completed_dates':len(completed),
            'total_dates':len(spec['feature_sessions']),'core_calls':stats['core_calls'],
            'partition_count':len(partitions)})
    for day_rows,evidence in _iter_stock_feature_days(data,config=config,catalog=catalog,chosen=chosen,
            qlib_inputs=prepared,stats=stats,history_sessions=history_length):
        day_bytes=_object_upper_bytes(day_rows)+_object_upper_bytes(evidence)
        require(day_bytes <= block_budget, 'one Feature date exceeds matrix storage resident/metadata budget')
        if proof and pending_bytes+day_bytes > block_budget: publish_block()
        rows.extend(day_rows); proof.append(evidence); pending_bytes+=day_bytes
        require([p['session'] for p in proof] == spec['feature_sessions'][len(completed):len(completed)+len(proof)],
                'prepared matrix Feature date order/coverage mismatch')
        # Codec allocation is a bounded row/column block, never a full-period
        # DataFrame. Native executor/proof RSS is measured by the caller too.
        require(len(rows)*min(len(schema),options['column_block'])*18 <= options['maximum_resident_bytes'],
                'matrix column buffer resident budget exceeded')
        if len(proof) < options['row_block_sessions'] and proof[-1]['session'] != spec['feature_sessions'][-1]: continue
        publish_block()
    require(completed == spec['feature_sessions'], 'incomplete matrix Feature coverage')
    # Admit every newly written block before publishing the COMPLETE index.
    # Progress callbacks and resumed files cannot turn a corrupt checkpoint
    # into a formal HIT that fails only after publication.
    from .stock_matrix_reader import VerifiedMatrixStore, admit_feature_matrix_parts
    with VerifiedMatrixStore(maximum_matrix_bytes=options['maximum_resident_bytes'],
            maximum_parent_bytes=DEFAULT_LIMITS['maximum_parent_bytes']) as store:
        covered,blocks,unused_sources,unused_parents=admit_feature_matrix_parts(store,
            spec=spec,view=view,schema=schema,row_index=row_wire,partitions=partitions,
            complete=True,universe_id=universe_id)
        require(covered == spec['feature_sessions'] and feature_rows == [r for block in blocks
                for r in block['_source_selection_rows']], 'complete Feature source-selection mismatch')
        store.check(); source_marks.update(store.fingerprints)
    selection=write_part(target,seal({'contract_version':'stock_feature_source_selection_v1',
        'definition_ref':definition_ref,'feature_rows':feature_rows},'source_selection_ref'),'source_selection_ref')
    _spec(spec)
    require(all(file_fingerprint(p) == mark for p,mark in source_marks.items()),
            'Feature scope/Qlib changed during preparation')
    value={'contract_version':'stock_feature_inputs_v2','definition':definition,'definition_ref':definition_ref,
        'status':'COMPLETE','qlib_view':view,'qlib_manifest':qdesc,'schema':schema,'schema_digest':digest(schema),
        'row_index':rid,'source_selection':selection,'partitions':partitions}
    value['feature_inputs_ref']=digest({k:value[k] for k in ('contract_version','definition_ref',
        'qlib_manifest','schema_digest','row_index','source_selection','partitions')})
    _atomic(target/'index.json',seal(value,'content_digest'))
    return load_stock_feature_inputs(target)


def build_stock_feature_inputs(data, *, spec, destination, shard_sessions=2, progress=None, storage_options=None):
    """Prepare original complete-date v1 parents with atomic per-shard resume.

    No Label, fitting, prediction, account or supplier call is made. Correctness
    never depends on retained trust: completed parents and Qlib files are verified
    before reuse. A changed definition selects a separate directory.
    """
    if storage_options is not None:
        require(shard_sessions == 2 and type(shard_sessions) is int,
                'v1 shard_sessions conflicts with matrix storage options')
        return _build_feature_matrix(data, spec=spec, destination=destination,
                                     storage_options=storage_options, progress=progress)
    from .feature_catalog import load_feature_catalog
    from .stock_ml import _implementation, _environment, _prepare_stock_qlib, _iter_stock_feature_days
    spec = deepcopy(spec)
    source_marks = {spec['scope']['path']:file_fingerprint(spec['scope']['path'])}
    universe_id = _spec(spec)
    require(type(shard_sessions) is int and shard_sessions > 0, 'positive shard session count required')
    catalog = load_feature_catalog(); chosen = catalog.select(spec['feature_selection'])
    require(catalog.identity == spec['catalog_ref'] and [c['id'] for c in chosen] == spec['ordered_features'],
            'current Feature catalog mismatch')
    history_length = max(c['lookback'] for c in chosen)
    calendar = spec['calendar']; first = calendar.index(spec['feature_sessions'][0])
    require(first >= history_length-1 and spec['read_sessions'] == calendar[
        first-history_length+1:calendar.index(spec['feature_sessions'][-1])+1], 'actual Feature history range mismatch')
    implementations = _implementation()
    definition = {'spec':spec,'shard_sessions':shard_sessions,'implementation_sources':implementations,
        'implementation_ref':digest(implementations),'environment':_environment()}
    require(all(file_fingerprint(p) == mark for p,mark in source_marks.items()),
            'Feature scope changed after admission')
    definition_ref = digest(definition); target = Path(destination)/definition_ref[7:]
    if (target/'index.json').exists():
        saved = load_stock_feature_inputs(target)
        require(saved.to_dict()['definition'] == definition, 'cached Feature input definition mismatch')
        return StockFeatureInputs(saved.path,True,saved.to_dict())
    target.mkdir(parents=True,exist_ok=True)
    config = {'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],'symbols':spec['universe'],
        'universe_id':universe_id,'calendar':calendar,'read_sessions':spec['read_sessions'],
        'feature_sessions':spec['feature_sessions'],'cutoff_by_session':spec['cutoff_by_session'],
        'feature_selection':spec['feature_selection'],'scope_ref':spec['scope']['scope_bundle_ref']}
    prepared = _prepare_stock_qlib(data,config=config,destination=target)
    view = prepared['view_reference']; _qlib_scope(view,spec,universe_id)
    qpath = prepared['qlib_path']/'axiom-qlib.json'
    qdesc = {'path':str(qpath.resolve()),'file_digest':file_digest(qpath),'view_id':view['view_id']}
    require(_qlib(qdesc,DEFAULT_LIMITS,fingerprints=source_marks) == view, 'prepared Qlib reference mismatch')
    checkpoint = target/'checkpoint.json'; parents = []; completed = []; marks = {}
    if checkpoint.exists():
        value = _read(checkpoint); _verify_ref(value,'content_digest')
        require(set(value) == {'contract_version','definition_ref','qlib_manifest','feature_parents','content_digest'} and
                value['contract_version'] == 'stock_feature_inputs_checkpoint_v1' and
                value['definition_ref'] == definition_ref and value['qlib_manifest'] == qdesc,
                'checkpoint input definition/Qlib mismatch')
        parents = value['feature_parents']
        completed,marks = _parents(parents,spec,view,universe_id,DEFAULT_LIMITS,complete=False)
        require(all(len(d['sessions']) == shard_sessions for d in parents[:-1]) and
                (not parents or len(parents[-1]['sessions']) == shard_sessions or completed == spec['feature_sessions']),
                'checkpoint partial shard mismatch')
    config['feature_sessions'] = spec['feature_sessions'][len(completed):]
    rows = []; proof = []; shard_days = []
    stats = {'data_read_calls':0,'core_calls':0,'feature_core_calls':0}
    for day_rows,evidence in _iter_stock_feature_days(data,config=config,catalog=catalog,chosen=chosen,
            qlib_inputs=prepared,stats=stats,history_sessions=history_length):
        rows.extend(day_rows); proof.append(evidence); shard_days.append(evidence['session'])
        if len(shard_days) < shard_sessions and shard_days[-1] != spec['feature_sessions'][-1]: continue
        require(shard_days == spec['feature_sessions'][len(completed):len(completed)+len(shard_days)],
                'prepared Feature date order/coverage mismatch')
        feature = seal({'contract_version':'stock_feature_build_v1','catalog_ref':spec['catalog_ref'],
            'selection':spec['feature_selection'],'ordered_features':spec['ordered_features'],
            'qlib_view':view,'input_evidence_ref':digest(proof),'rows':rows},'feature_ref')
        published = target/'parents'/feature['feature_ref'][7:]; published.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.feature-',dir=published.parent) as temporary:
            stage = Path(temporary); write_json(stage/'features.json',feature);write_json(stage/'feature-inputs.json',proof)
            feature_hash = file_digest(stage/'features.json'); proof_hash = file_digest(stage/'feature-inputs.json')
            def descriptor(root):
                return {'features':{'path':str((root/'features.json').resolve()),'file_digest':feature_hash,
                                    'feature_ref':feature['feature_ref']},
                    'input_evidence':{'path':str((root/'feature-inputs.json').resolve()),
                        'file_digest':proof_hash,'input_evidence_ref':feature['input_evidence_ref']},
                    'sessions':list(shard_days)}
            local = descriptor(stage)
            _parents([local],spec,view,universe_id,DEFAULT_LIMITS,complete=True,expected_dates=list(shard_days))
            try: stage.rename(published)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
            desc = descriptor(published)
            # Validate a concurrently published/orphaned directory before reuse.
            feature_check, indexed, by_day = validate_feature_parent(desc,spec)
            del feature_check,indexed,by_day
        parents.append(desc); completed.extend(shard_days)
        _atomic(checkpoint,seal({'contract_version':'stock_feature_inputs_checkpoint_v1',
            'definition_ref':definition_ref,'qlib_manifest':qdesc,'feature_parents':parents},'content_digest'))
        rows=[];proof=[];shard_days=[];del feature
        if progress is not None:
            progress({'stage':'feature_parent','completed_dates':len(completed),
                      'total_dates':len(spec['feature_sessions']),'core_calls':stats['core_calls']})
    require(all(file_fingerprint(p) == mark for p,mark in marks.items()), 'resumed parent changed during preparation')
    _parents(parents,spec,view,universe_id,DEFAULT_LIMITS,complete=True)
    _spec(spec)
    require(all(file_fingerprint(p) == mark for p,mark in source_marks.items()),
            'Feature scope/Qlib changed during preparation')
    value = {'contract_version':'stock_feature_inputs_v1','definition':definition,'definition_ref':definition_ref,
        'status':'COMPLETE','qlib_view':view,'qlib_manifest':qdesc,'feature_parents':parents}
    value['feature_inputs_ref'] = digest({k:value[k] for k in ('definition_ref','qlib_manifest','feature_parents')})
    _atomic(target/'index.json',seal(value,'content_digest'))
    return load_stock_feature_inputs(target)

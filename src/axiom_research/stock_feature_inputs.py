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


def _spec(spec):
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
    scope = read_parent(spec['scope'], 'scope_bundle_ref')
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


def _qlib(descriptor, limits, *, fingerprints=None):
    """Verify the saved Data manifest and its actual files, without importing Data."""
    require(type(descriptor) is dict and set(descriptor) == {'path','file_digest','view_id'} and
            Path(descriptor['path']).is_absolute(), 'explicit saved Qlib manifest required')
    path = Path(descriptor['path']); _bounded(path, limits)
    marks = {str(path):file_fingerprint(path)}
    require(file_digest(path) == descriptor['file_digest'], 'Qlib manifest file mismatch')
    manifest = _read(path)
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
                file_digest(child) == 'sha256:'+expected['sha256'], 'Qlib file mismatch: '+relative)
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


def _parents(parents, spec, view, universe_id, limits, *, complete, expected_dates=None):
    """Same v1 validator plus the new exact-scope source closure; one pair resident."""
    covered = []; marks = {}
    history_length = spec['read_sessions'].index(spec['feature_sessions'][0]) + 1
    expected_dates = spec['feature_sessions'] if expected_dates is None else expected_dates
    for desc in parents:
        for name in ('features','input_evidence'):
            path = desc[name]['path']; _bounded(path, limits); marks[path] = file_fingerprint(path)
        require(desc['sessions'] == expected_dates[len(covered):len(covered)+len(desc['sessions'])],
                'parent date order/coverage mismatch')
        feature, indexed, proof = validate_feature_parent(desc, spec)
        require(feature['qlib_view'] == view, 'parent Qlib identity mismatch')
        for day in desc['sessions']:
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
            sources = p['source_evidence']; require(bool(sources), 'Feature source proof required')
            membership = False
            for source in sources.values():
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
            require(membership, 'Feature membership source missing')
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


def build_stock_feature_inputs(data, *, spec, destination, shard_sessions=2, progress=None):
    """Prepare original complete-date v1 parents with atomic per-shard resume.

    No Label, fitting, prediction, account or supplier call is made. Correctness
    never depends on retained trust: completed parents and Qlib files are verified
    before reuse. A changed definition selects a separate directory.
    """
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

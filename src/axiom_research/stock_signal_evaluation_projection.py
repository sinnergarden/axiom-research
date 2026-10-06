"""Frozen, date-sharded evaluation inputs. No Data, ML or numeric imports."""
from copy import deepcopy
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile

from .api import from_dict, semantic_identity, to_dict, validate
from .contracts import ArtifactRef
from .stock_artifacts import digest, file_digest, write_json, _verify_ref
from .stock_label_contracts import _instant, _finite
from .stock_signal_evaluation_inputs import (
    _require, _scope, _admit_inputs, _select_inputs, _source_records)


INPUT_VERSION = 'stock_signal_evaluation_inputs_v1'
ROOT_FIELDS = {'contract_version', 'input_id', 'scope', 'signal_order', 'signal_refs',
    'signal_metadata', 'raw_metadata', 'clock_floor', 'admission_receipt', 'shards'}
LABEL_FIELDS = {'security_id', 'feature_session', 'start_session', 'end_session',
    'return', 'valid', 'invalid_reason', 'label_available_at', 'source_refs'}


def _mark(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _check_marks(marks):
    for path, mark in marks.items():
        _require(_mark(path.stat()) == mark, 'frozen evaluation file changed: ' + str(path))


def _read_checked(path, expected=None, *, marks=None):
    """Hash and parse the exact buffer consumed, including duplicate-key checks."""
    path = Path(path)
    with path.open('rb') as stream:
        before = _mark(os.fstat(stream.fileno()))
        payload = stream.read()
        _require(_mark(os.fstat(stream.fileno())) == before,
                 'frozen evaluation file changed during read: ' + str(path))
    _require(_mark(path.stat()) == before, 'frozen evaluation file replaced: ' + str(path))
    observed = 'sha256:' + sha256(payload).hexdigest()
    _require(expected is None or observed == expected, 'frozen evaluation file digest mismatch: ' + str(path))
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    value = json.loads(payload, object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if marks is not None:
        _require(path not in marks or marks[path] == before, 'frozen evaluation file changed between reads')
        marks[path] = before
    return value, observed


def _input_ref(value):
    value = from_dict(value) if type(value) is dict else value
    _require(type(value) is ArtifactRef, 'frozen evaluation ArtifactRef required')
    validate(value)
    _require(value.artifact_type == 'StockSignalEvaluationInputs' and
        value.artifact_contract_version == INPUT_VERSION, 'frozen evaluation input ref type/version mismatch')
    _require(Path(value.uri).is_absolute(), 'fixed absolute frozen input locator required')
    return value


def _root_id(root):
    body = {k: v for k, v in root.items() if k != 'input_id'}
    # The original absolute locations remain useful for audit; identities bind
    # their immutable content descriptors instead of the locations themselves.
    body = deepcopy(body)
    receipt = body['admission_receipt']
    receipt.pop('receipt_ref', None)
    def locations(item):
        if type(item) is dict:
            return {k: locations(v) for k, v in item.items() if k != 'path'}
        if type(item) is list:
            return [locations(v) for v in item]
        return item
    body['admission_receipt'] = locations(receipt)
    return digest(body)


def _records(closures):
    records = {}
    for sources in closures.values():
        for source in sources:
            for record in source['source_records']:
                path = Path(record['path'])
                _require(path not in records or records[path] == record['file_digest'],
                         'conflicting frozen source descriptors')
                records[path] = record['file_digest']
    return records


def _compact_closure(closures):
    records = _records(closures)
    table = [{'path': str(path), 'file_digest': records[path]} for path in sorted(records, key=str)]
    indices = {row['path']: i for i, row in enumerate(table)}
    compact = {name: [{**{k: deepcopy(v) for k, v in source.items() if k != 'source_records'},
        'source_record_indices': [indices[row['path']] for row in source['source_records']]}
        for source in sources] for name, sources in closures.items()}
    return compact, table


def _expand_closure(receipt):
    table = receipt['source_records']
    _require(type(table) is list and all(type(r) is dict and set(r) == {'path', 'file_digest'} for r in table) and
        [r['path'] for r in table] == sorted(set(r['path'] for r in table)), 'frozen unique source records required')
    expanded = {}
    for name, sources in receipt['source_closure'].items():
        expanded[name] = []
        for source in sources:
            _require(set(source) == {'signal_input', 'model_ref', 'feature_ref', 'source_record_indices'} and
                type(source['source_record_indices']) is list and
                all(type(i) is int and 0 <= i < len(table) for i in source['source_record_indices']),
                'frozen source record indices mismatch')
            expanded[name].append({**{k: v for k, v in source.items() if k != 'source_record_indices'},
                'source_records': [table[i] for i in source['source_record_indices']]})
    return expanded


def _source_pins(signal_inputs, raw_label_input):
    records = {}
    seen = set()
    _require(all(type(name) is str and name for name in signal_inputs), 'ordered Signal names required')
    _require(type(raw_label_input) is dict and set(raw_label_input) == {'path', 'file_digest', 'label_ref'} and
        type(raw_label_input['path']) is str and Path(raw_label_input['path']).is_absolute(),
        'explicit fixed Raw Label descriptor required')
    for inputs in signal_inputs.values():
        for descriptor in inputs if type(inputs) is list else [inputs]:
            _require(type(descriptor) is dict and set(descriptor) == {'path', 'file_digest', 'signal_run_ref'} and
                type(descriptor['path']) is str and Path(descriptor['path']).is_absolute() and
                Path(descriptor['path']).name == 'predictions.json', 'explicit fixed Signal descriptor required')
            root = Path(descriptor['path']).parent
            if root in seen:
                continue
            seen.add(root)
            for row in _source_records(root):
                path = Path(row['path'])
                _require(path not in records or records[path] == row['file_digest'], 'conflicting source pins')
                records[path] = row['file_digest']
            _require(records.get(Path(descriptor['path'])) == descriptor['file_digest'], 'saved Signal pin mismatch')
    path = Path(raw_label_input['path'])
    _require(path not in records or records[path] == raw_label_input['file_digest'], 'conflicting Raw Label pin')
    records[path] = raw_label_input['file_digest']
    return records, {path: _mark(path.stat()) for path in records}


def _raw_metadata(raw):
    return {**{k: deepcopy(raw[k]) for k in
        ('contract_version', 'label_ref', 'label_spec', 'calendar_ref', 'source_ref')},
        'source_context': deepcopy(raw['source_evidence']['context'])}


def _clock_floor(metadata, raw):
    clocks = [m['evaluation_clock_floor'] for items in metadata.values() for m in items]
    context = raw['source_context']
    for query in [context['query'], context['derivation']['price_query'], context['derivation']['factor_query']]:
        clocks.extend(query['cutoff_by_session'].values())
    return max(map(_instant, clocks)).isoformat().replace('+00:00', 'Z')


def _shard(admission, scope, day):
    rows = []
    for security in scope['universe']:
        key = security, day
        membership = {item['members'][key]['member'] for item in admission['projected'].values()
                      if key in item['members']}
        _require(len(membership) <= 1, 'comparison historical membership conflict')
        member = next(iter(membership)) if membership else None
        label = admission['labels'][key]
        predictions = {}
        for name, item in admission['projected'].items():
            prediction = deepcopy(item['rows'].get(key))
            if prediction is not None:
                feature = item['members'][key]
                clocks = [_instant(c) for c in feature['availability'] if c is not None]
                prediction['feature_knowledge_cutoff'] = feature['knowledge_cutoff']
                prediction['feature_available_at'] = (max(clocks).isoformat().replace('+00:00', 'Z') if clocks else None)
            predictions[name] = prediction
        rows.append({'security_id': security, 'member': member,
            'label': {k: deepcopy(label.get(k)) for k in LABEL_FIELDS},
            'predictions': predictions})
    return {'contract_version': 'stock_signal_evaluation_date_v1', 'session': day, 'rows': rows}


def save_stock_signal_evaluation_inputs(signal_inputs, *, raw_label_input, scope, destination):
    """Fully admit original sources, then atomically freeze the small projection."""
    scope = _scope(scope)
    _require(type(signal_inputs) is dict and bool(signal_inputs), 'ordered Signal mapping required')
    pins, source_marks = _source_pins(signal_inputs, raw_label_input)
    admitted = _admit_inputs(signal_inputs, raw_label_input, scope)
    observed = _records(admitted['closures'])
    _require(all(pins.get(path) == ref for path, ref in observed.items()), 'original source changed during admission')
    _require(file_digest(raw_label_input['path']) == raw_label_input['file_digest'], 'Raw Label changed during admission')
    _check_marks(source_marks)
    # Exercise exactly the same masks/coverage checks before publishing.
    _select_inputs(admitted, scope)
    raw = _raw_metadata(admitted['raw'])
    closure, records = _compact_closure(admitted['closures'])
    receipt = {'contract_version': 'stock_signal_evaluation_admission_v1',
        'signal_inputs': deepcopy(signal_inputs), 'raw_label_input': deepcopy(raw_label_input),
        'scope': scope, 'source_closure': closure, 'source_records': records,
        'validation_sources': {name: file_digest(Path(__file__).parent/name) for name in
            ('stock_signal_evaluation_projection.py', 'stock_signal_evaluation_inputs.py',
             'stock_artifacts.py', 'stock_label_contracts.py', 'stock_fold_artifacts.py',
             'stock_fold_inputs.py', 'stock_training.py')}}
    receipt['receipt_ref'] = digest(receipt)
    root = {'contract_version': INPUT_VERSION, 'scope': scope, 'signal_order': list(signal_inputs),
        'signal_refs': admitted['refs'], 'signal_metadata': admitted['metadata'],
        'raw_metadata': raw, 'clock_floor': _clock_floor(admitted['metadata'], raw),
        'admission_receipt': receipt, 'shards': {}}
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.signal-inputs-', dir=destination) as temporary:
        stage = Path(temporary)/'complete'; stage.mkdir()
        for day in scope['sessions']:
            name = day + '.json'
            write_json(stage/name, _shard(admitted, scope, day))
            root['shards'][day] = {'file': name, 'file_digest': file_digest(stage/name)}
        root['input_id'] = _root_id(root)
        write_json(stage/'manifest.json', root)
        ref = ArtifactRef(artifact_type='StockSignalEvaluationInputs', artifact_id=root['input_id'],
            artifact_contract_version=INPUT_VERSION, content_digest=file_digest(stage/'manifest.json'),
            uri=str(stage/'manifest.json'))
        _load_inputs(ref, scope)
        _check_marks(source_marks)
        target = destination/root['input_id'][7:]
        final = ArtifactRef(artifact_type=ref.artifact_type, artifact_id=ref.artifact_id,
            artifact_contract_version=ref.artifact_contract_version, content_digest=ref.content_digest,
            uri=str(target/'manifest.json'))
        if target.exists():
            _load_inputs(final, scope)
            return final
        try:
            os.rename(stage, target)
        except OSError:
            if not target.exists():
                raise
            _load_inputs(final, scope)
        return final


def _verify_root(root, ref, scope):
    _require(type(root) is dict and set(root) == ROOT_FIELDS and root['contract_version'] == INPUT_VERSION,
             'frozen input root contract mismatch')
    _require(root['input_id'] == ref.artifact_id == _root_id(root), 'frozen input identity mismatch')
    base = _scope(root['scope'])
    _require(base == root['scope'] and scope['calendar'] == base['calendar'] and
        set(scope['sessions']) <= set(base['sessions']) and set(scope['universe']) <= set(base['universe']),
        'evaluation outside frozen input scope/calendar')
    names = root['signal_order']
    _require(type(names) is list and bool(names) and all(type(n) is str and n for n in names) and
        len(names) == len(set(names)) and set(names) == set(root['signal_refs']) == set(root['signal_metadata']),
        'frozen comparison axes mismatch')
    receipt = root['admission_receipt']; _verify_ref(receipt, 'receipt_ref')
    _require(set(receipt) == {'contract_version', 'signal_inputs', 'raw_label_input', 'scope',
        'source_closure', 'source_records', 'validation_sources', 'receipt_ref'} and
        receipt['contract_version'] == 'stock_signal_evaluation_admission_v1' and receipt['scope'] == base and
        set(receipt['signal_inputs']) == set(names) == set(receipt['source_closure']), 'frozen admission receipt mismatch')
    expanded = _expand_closure(receipt)
    raw = root['raw_metadata']
    _require(set(raw) == {'contract_version', 'label_ref', 'label_spec', 'calendar_ref', 'source_ref', 'source_context'} and
        raw['contract_version'] == 'stock_label_build_v1' and
        raw['label_ref'] == receipt['raw_label_input']['label_ref'] and raw['calendar_ref'] == digest({
            'contract_version': 'stock_label_calendar_v1', 'sessions': base['calendar']}), 'frozen Raw Label binding mismatch')
    spec = raw['label_spec']
    _require(spec.get('normalization') == 'none' and type(spec.get('horizon_sessions')) is int and
        spec['horizon_sessions'] > 0 and type(spec.get('start_session_offset')) is int and
        spec['start_session_offset'] == 1 and type(spec.get('end_session_offset')) is int and
        spec['end_session_offset'] == spec['horizon_sessions'] and
        spec.get('label_id') == f"forward_{spec['horizon_sessions']}_session_open_close_v1" and
        spec.get('start_price') == 'open' and spec.get('end_price') == 'close', 'frozen Raw Label semantics mismatch')
    context = raw['source_context']
    by_day = {name: {} for name in names}
    for name in names:
        metadata = root['signal_metadata'][name]
        sources = expanded[name]
        _require(type(metadata) is list and len(metadata) == len(sources) and bool(metadata) and
            [m['signal_run_ref'] for m in metadata] == root['signal_refs'][name] ==
            [s['signal_input']['signal_run_ref'] for s in sources], 'frozen Signal version binding mismatch')
        descriptors = receipt['signal_inputs'][name]
        descriptors = descriptors if type(descriptors) is list else [descriptors]
        _require([s['signal_input'] for s in sources] == descriptors, 'frozen Signal descriptor mismatch')
        for meta, source in zip(metadata, sources):
            _verify_ref(meta['model'], 'model_ref')
            _require(meta['model']['model_ref'] == source['model_ref'] and meta['feature_ref'] == source['feature_ref'] and
                meta['signal_contract_version'] in ('stock_prediction_run_v1', 'stock_prediction_run_v2') and
                meta['signal_stage'] == 'prediction_raw' and meta['score_unit'] == 'dimensionless' and
                meta['score_semantics'] == meta['model']['target_semantics'] and
                meta['snapshot'] == context['snapshot_id'] and meta['pit_policy'] == context['query']['pit_policy'],
                'frozen Signal model/stage/Snapshot/PIT mismatch')
            dates = meta['prediction_sessions']
            _require(type(dates) is list and bool(dates) and dates == sorted(set(dates)) and
                set(dates) <= set(base['calendar']), 'frozen Signal ordered date coverage mismatch')
            for day in dates:
                _require(day not in by_day[name], 'frozen weekly Signal date overlap')
                by_day[name][day] = meta
    _require(root['clock_floor'] == _clock_floor(root['signal_metadata'], raw) and
        _instant(scope['evaluation_cutoff']) >= _instant(root['clock_floor']),
        'evaluation cutoff precedes frozen source revision/Signal visibility')
    _require(set(root['shards']) == set(base['sessions']), 'complete frozen date shards required')
    for day, descriptor in root['shards'].items():
        _require(set(descriptor) == {'file', 'file_digest'} and descriptor['file'] == day + '.json',
                 'frozen shard locator mismatch')
    return by_day


def _load_inputs(input_ref, scope, *, marks=None):
    ref = _input_ref(input_ref); scope = _scope(scope)
    marks = {} if marks is None else marks
    root, _ = _read_checked(ref.uri, ref.content_digest, marks=marks)
    metadata_by_day = _verify_root(root, ref, scope)
    names, base = root['signal_order'], root['scope']
    shared_members, labels = {}, {}
    projected = {name: {'rows': {}, 'members': shared_members} for name in names}
    positions = {day: i for i, day in enumerate(base['calendar'])}
    wanted = set(scope['universe'])
    cutoff = _instant(scope['evaluation_cutoff'])
    raw = root['raw_metadata']; raw_cutoff = _instant(raw['source_context']['derivation']['decision_cutoff'])
    for day in scope['sessions']:
        descriptor = root['shards'][day]
        shard, _ = _read_checked(Path(ref.uri).parent/descriptor['file'], descriptor['file_digest'], marks=marks)
        _require(set(shard) == {'contract_version', 'session', 'rows'} and
            shard['contract_version'] == 'stock_signal_evaluation_date_v1' and shard['session'] == day and
            type(shard['rows']) is list and [r['security_id'] for r in shard['rows']] == base['universe'],
            'frozen shard complete ordered keys required')
        for row in shard['rows']:
            _require(set(row) == {'security_id', 'member', 'label', 'predictions'} and
                (row['member'] is None or type(row['member']) is bool) and set(row['predictions']) == set(names),
                'frozen membership/prediction fields mismatch')
            key = row['security_id'], day
            label = row['label']
            _require(set(label) == LABEL_FIELDS and (label['security_id'], label['feature_session']) == key and
                type(label['valid']) is bool and label['source_refs'] == [raw['source_ref']], 'frozen Label key/source mismatch')
            for field, offset in (('start_session', 1), ('end_session', raw['label_spec']['horizon_sessions'])):
                i = positions[day] + offset
                _require(label[field] == (base['calendar'][i] if i < len(base['calendar']) else None),
                         'frozen Label calendar endpoint mismatch')
            if label['valid']:
                _require(_finite(label['return']) and label['invalid_reason'] is None and
                    label['start_session'] is not None and label['end_session'] is not None and
                    _instant(label['label_available_at']) <= raw_cutoff, 'frozen valid Label value/clock mismatch')
            else:
                _require(label['return'] is None and bool(label['invalid_reason']), 'frozen invalid Label null/reason mismatch')
            for name, prediction in row['predictions'].items():
                if prediction is None:
                    continue
                _require((prediction['security_id'], prediction['session']) == key and
                    type(prediction['valid']) is bool and type(prediction['member']) is bool and
                    prediction['member'] is row['member'], 'frozen Signal key/member mismatch')
                knowledge, available = _instant(prediction['knowledge_cutoff']), _instant(prediction['available_at'])
                _require(available <= knowledge <= cutoff and knowledge.date().isoformat() >= day,
                         'frozen Signal source clock conflict')
                feature_knowledge = _instant(prediction['feature_knowledge_cutoff'])
                feature_available = prediction['feature_available_at']
                _require(feature_knowledge <= knowledge and
                    (feature_available is None or _instant(feature_available) <= feature_knowledge),
                    'frozen Feature dependency clock mismatch')
                meta = metadata_by_day[name].get(day)
                _require(meta is not None and meta['model']['model_ref'] in prediction['source_refs'] and
                    meta['feature_ref'] in prediction['source_refs'], 'frozen Signal original model/Feature mismatch')
                _require(_instant(meta['evaluation_clock_floor']) >= knowledge,
                         'frozen Signal admission clock floor mismatch')
                _require(_instant(meta['model']['fit_cutoff']) <= available, 'frozen model fit clock mismatch')
                if meta['signal_contract_version'] == 'stock_prediction_run_v2':
                    model_clock = meta['model']['simulated_available_at']
                    _require(prediction['simulated_model_available_at'] == model_clock and
                        _instant(model_clock) < available and _instant(prediction['feature_knowledge_cutoff']) <= knowledge,
                        'frozen model publication/inference clock mismatch')
                else:
                    dependencies = [_instant(meta['model']['fit_cutoff'])]
                    if feature_available is not None:
                        dependencies.append(_instant(feature_available))
                    _require(available == max(dependencies) and knowledge == feature_knowledge,
                             'frozen original prediction dependency clock mismatch')
                if prediction['valid']:
                    _require(row['member'] is True and _finite(prediction['score']) and prediction['invalid_reason'] is None,
                             'frozen valid Signal value mismatch')
                else:
                    _require(prediction['score'] is None and bool(prediction['invalid_reason']), 'frozen invalid Signal null/reason mismatch')
                if key[0] in wanted:
                    projected[name]['rows'][key] = prediction
            if key[0] in wanted:
                labels[key] = label; shared_members[key] = {'member': row['member']}
    _check_marks(marks)
    admission = {'projected': projected, 'labels': labels, 'raw': raw,
        'refs': {name: root['signal_refs'][name] for name in names}, 'closures': _expand_closure(root['admission_receipt'])}
    return ref, root, _select_inputs(admission, scope)


def _audit_input(input_ref):
    ref = _input_ref(input_ref)
    root, _ = _read_checked(ref.uri, ref.content_digest)
    marks = {}
    _, root, _ = _load_inputs(ref, root['scope'], marks=marks)
    receipt = root['admission_receipt']
    # Hash the full original source set, including unused training ancestors.
    records = _records(_expand_closure(receipt))
    records[Path(receipt['raw_label_input']['path'])] = receipt['raw_label_input']['file_digest']
    source_marks = {path: _mark(path.stat()) for path in records}
    for path, expected in records.items():
        _require(file_digest(path) == expected, 'audit original source mismatch: ' + str(path))
    _require(file_digest(receipt['raw_label_input']['path']) == receipt['raw_label_input']['file_digest'],
             'audit original Raw Label mismatch')
    original = _admit_inputs({name: receipt['signal_inputs'][name] for name in root['signal_order']},
                            receipt['raw_label_input'], root['scope'])
    _require(original['closures'] == _expand_closure(receipt) and original['metadata'] == root['signal_metadata'] and
        _raw_metadata(original['raw']) == root['raw_metadata'], 'audit original metadata/projection mismatch')
    for day, descriptor in root['shards'].items():
        shard, _ = _read_checked(Path(ref.uri).parent/descriptor['file'], descriptor['file_digest'], marks=marks)
        _require(_shard(original, root['scope'], day) == shard, 'audit original date projection mismatch: ' + day)
    _check_marks(source_marks); _check_marks(marks)
    return ref


def _implementation_v3():
    module = import_module('axiom_engine.core.signal_statistics')
    core = Path(module.__file__).parent
    names = ('stock_signal_evaluation.py', 'stock_signal_evaluation_inputs.py',
             'stock_artifacts.py', 'stock_label_contracts.py', 'stock_signal_evaluation_projection.py')
    return {'research': {name: file_digest(Path(__file__).parent/name) for name in names},
        'core': {name: file_digest(core/name) for name in ('signal_statistics.py', 'contracts.py')},
        'operator': 'axiom_engine.core.evaluate_signal_statistics'}


def _group_key(group):
    return digest({'contract_version': 'stock_signal_evidence_v3',
        'input_identity': semantic_identity(_input_ref(group['input_ref'])),
        **{key: group[key] for key in ('scope', 'signal_order', 'spec_ref', 'sample_mask_ref',
            'common_input_ref', 'native_input_ref', 'implementation_ref')}})


def _group_definition(ref, root, admitted, implementation):
    from .stock_signal_evaluation import SPEC
    group = {'contract_version': 'stock_signal_evaluation_group_v1',
        'input_ref': to_dict(ref), 'admission_receipt_ref': root['admission_receipt']['receipt_ref'],
        'scope': admitted['scope'], 'signal_order': list(admitted['signal_keys']),
        'spec_ref': digest(SPEC), 'spec': deepcopy(SPEC), 'sample_mask_ref': digest(admitted['mask']),
        'common_input_ref': digest(admitted['common_input']), 'native_input_ref': digest(admitted['native_input']),
        'implementation_ref': digest(implementation), 'implementation_sources': implementation}
    group['evaluation_key'] = _group_key(group)
    return group


def _reports_v3(group, admitted, common, native):
    reports = {}
    counts = Counter(day for _, day in admitted['common_keys'])
    for name, key in admitted['signal_keys'].items():
        series = [r for r in common['series'] if r['signal_key'] == key]
        summary = next(r for r in common['summary'] if r['signal_key'] == key)
        own_native = {**deepcopy(admitted['native_coverage'][name]),
            'statistics_input_ref': native['input_ref'], 'statistics_ref': native['statistics_ref'],
            'series': [r for r in native['series'] if r['signal_key'] == key],
            'summary': next(r for r in native['summary'] if r['signal_key'] == key)}
        coverage = deepcopy(admitted['native_coverage'][name])
        coverage['valid_pair_count'] = len(admitted['common_keys'])
        coverage['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = own_native['valid_pair_count'] - len(admitted['common_keys'])
        for row in coverage['by_session']:
            row['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = row['valid_pair_count'] - counts[row['session']]
            row['valid_pair_count'] = counts[row['session']]
        coverage.update(comparison_mode='common_valid_key_intersection', native=own_native,
            common_statistics_ref=common['statistics_ref'], native_statistics_ref=native['statistics_ref'])
        report = {'contract_version': 'stock_signal_evidence_v3',
            'validation_basis': 'frozen_projection_v1', 'evaluation_key': group['evaluation_key'],
            'input_signal_refs': admitted['refs'][name], 'input_evidence': {
                'input_ref': group['input_ref'], 'signal_name': name,
                'admission_receipt_ref': group['admission_receipt_ref']},
            'label_ref': admitted['raw']['label_ref'], 'label_spec': deepcopy(admitted['raw']['label_spec']),
            'scope': group['scope'], 'spec': group['spec'], 'spec_ref': group['spec_ref'],
            'sample_mask_ref': group['sample_mask_ref'], 'statistics_input_ref': common['input_ref'],
            'statistics_ref': common['statistics_ref'], 'series': series, 'summary': summary,
            'coverage': coverage, 'status': 'COMPLETE' if summary['valid_ic_session_count'] else 'NO_VALID_SESSIONS',
            'limitations': ['Forward-label statistics are not account returns.',
                'Overlapping label horizons and short-sample IR are descriptive, without a confidence claim.',
                'Source closure was checked during freezing; this load verifies the frozen projection. Full audit is explicit.',
                'Label spec is copied from the saved Raw Label build; absent unit fields remain absent.'],
            'implementation_ref': group['implementation_ref']}
        report['evidence_ref'] = digest({'evaluation_key': group['evaluation_key'], 'signal_name': name})
        report['content_digest'] = digest(report)
        reports[name] = report
    return reports


def _load_group(path, *, prepared=None, marks=None, expected_key=None, reused=False):
    from .stock_signal_evaluation import SPEC, _statistics, _verified_evaluation
    path = Path(path); marks = {} if marks is None else marks
    manifest, _ = _read_checked(path/'manifest.json', marks=marks)
    _require(set(manifest) == {'contract_version', 'evaluation_key', 'files'} and
        manifest['contract_version'] == 'stock_signal_evaluation_manifest_v2', 'frozen evaluation group manifest mismatch')
    files = manifest['files']
    _require(type(files) is dict and 'group.json' in files, 'complete frozen evaluation group required')
    group, _ = _read_checked(path/'group.json', files['group.json'], marks=marks)
    _require(set(group) == {'contract_version', 'input_ref', 'admission_receipt_ref', 'scope', 'signal_order',
        'spec_ref', 'spec', 'sample_mask_ref', 'common_input_ref', 'native_input_ref', 'implementation_ref',
        'implementation_sources', 'evaluation_key'} and group['contract_version'] == 'stock_signal_evaluation_group_v1' and
        group['spec'] == SPEC and group['spec_ref'] == digest(SPEC) and
        group['implementation_ref'] == digest(group['implementation_sources']) and
        group['evaluation_key'] == manifest['evaluation_key'] == _group_key(group), 'frozen evaluation group binding mismatch')
    _require(expected_key is None or group['evaluation_key'] == expected_key, 'frozen evaluation HIT identity mismatch')
    prepared = _load_inputs(group['input_ref'], group['scope'], marks=marks) if prepared is None else prepared
    ref, root, admitted = prepared
    stored_ref = _input_ref(group['input_ref'])
    _require(semantic_identity(ref) == semantic_identity(stored_ref), 'frozen evaluation input identity mismatch')
    expected = _group_definition(stored_ref, root, admitted, group['implementation_sources'])
    _require(group == expected, 'frozen evaluation exact input/mask binding mismatch')
    expected_files = {'group.json', 'common-statistics.json', 'native-statistics.json'}
    expected_files.update(digest({'evaluation_key': group['evaluation_key'], 'signal_name': name})[7:]+'.json'
                          for name in group['signal_order'])
    _require(set(files) == expected_files, 'frozen evaluation missing/extra group files')
    common, _ = _read_checked(path/'common-statistics.json', files['common-statistics.json'], marks=marks)
    native, _ = _read_checked(path/'native-statistics.json', files['native-statistics.json'], marks=marks)
    _statistics(common, admitted['common_input'], SPEC)
    _statistics(native, admitted['native_input'], SPEC)
    reports = _reports_v3(group, admitted, common, native)
    results = {}
    for name, report in reports.items():
        filename = report['evidence_ref'][7:]+'.json'
        actual, _ = _read_checked(path/filename, files[filename], marks=marks)
        _require(actual == report, 'frozen evaluation saved report mismatch: ' + name)
        results[name] = _verified_evaluation(path/filename, reused, actual)
    _check_marks(marks)
    return results


def _load_v3_report(path):
    path = Path(path)
    results = _load_group(path if path.is_dir() else path.parent)
    if path.is_dir():
        _require(len(results) == 1, 'select a report file for a multi-Signal evaluation group')
        return next(iter(results.values()))
    candidates = [saved for saved in results.values() if saved.path == path]
    _require(len(candidates) == 1, 'saved file is not a frozen Signal report')
    return candidates[0]


def evaluate_stock_signal_inputs(input_ref, *, scope, destination):
    """Verify frozen inputs and return an exact saved group HIT before Core."""
    from .stock_signal_evaluation import SPEC, _verified_evaluation
    marks = {}
    prepared = _load_inputs(input_ref, scope, marks=marks)
    ref, root, admitted = prepared
    group = _group_definition(ref, root, admitted, _implementation_v3())
    destination = Path(destination).resolve(); target = destination/group['evaluation_key'][7:]
    if target.exists():
        return _load_group(target, prepared=prepared, marks=marks, expected_key=group['evaluation_key'], reused=True)
    from axiom_engine.core import evaluate_signal_statistics
    common = evaluate_signal_statistics(admitted['common_input'], spec=SPEC).to_dict()
    native = evaluate_signal_statistics(admitted['native_input'], spec=SPEC).to_dict()
    reports = _reports_v3(group, admitted, common, native)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.signal-group-', dir=destination) as temporary:
        stage = Path(temporary)/'complete'; stage.mkdir()
        documents = {'group.json': group, 'common-statistics.json': common, 'native-statistics.json': native,
            **{report['evidence_ref'][7:]+'.json': report for report in reports.values()}}
        for name, document in documents.items():
            write_json(stage/name, document)
        write_json(stage/'manifest.json', {'contract_version': 'stock_signal_evaluation_manifest_v2',
            'evaluation_key': group['evaluation_key'], 'files': {name: file_digest(stage/name) for name in documents}})
        _load_group(stage, prepared=prepared, expected_key=group['evaluation_key'])
        _check_marks(marks)
        try:
            os.rename(stage, target)
        except OSError:
            if not target.exists():
                raise
            winner = _load_group(target, prepared=prepared, marks=marks, expected_key=group['evaluation_key'], reused=True)
            _require({name: saved.to_dict() for name, saved in winner.items()} == reports,
                     'immutable frozen evaluation collision')
            return winner
    return {name: _verified_evaluation(target/(report['evidence_ref'][7:]+'.json'), False, report)
            for name, report in reports.items()}


def audit_stock_signal_evaluation(path):
    """Revalidate every original source and compare its exact frozen projection."""
    saved = _load_v3_report(path)
    _audit_input(saved.to_dict()['input_evidence']['input_ref'])
    return saved

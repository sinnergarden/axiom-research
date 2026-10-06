"""Keyed OOS evaluation admission for already verified matrix batches.

The matrix owner supplies the OOS lease and saved-fold admission. This module
only joins those original targets; all statistics remain in the existing Core.
"""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import os
import tempfile

from .stock_artifacts import digest, file_digest, write_json, _verify_ref
from .stock_label_contracts import _instant, _finite
from .stock_signal_evaluation_inputs import (_grid, _require, _select_inputs,
    _validate_raw_labels, _validate_raw_label_header)


TARGET_FIELDS = {'security_id', 'feature_session', 'start_session', 'end_session',
    'return', 'valid', 'invalid_reason', 'label_available_at', 'source_refs'}
SLICE_FIELDS = {'contract_version', 'prepared_view_ref', 'fold_spec_ref',
    'selector', 'cutoff', 'rows', 'label_ref'}
RAW_FIELDS = {'raw_input', 'contract_version', 'label_ref', 'label_spec',
    'calendar_ref', 'source_ref', 'source_context', 'sessions'}


def _ref(value):
    return (type(value) is str and len(value) == 71 and value.startswith('sha256:')
            and all(c in '0123456789abcdef' for c in value[7:]))


def _copy_wire(value):
    if isinstance(value, Mapping):
        return {k: _copy_wire(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_copy_wire(v) for v in value]
    return deepcopy(value)


def _label_semantics(spec):
    _require(isinstance(spec, Mapping), 'matrix evaluation RawLabel spec required')
    horizon = spec.get('horizon_sessions')
    _require(type(horizon) is int and horizon > 0 and spec.get('normalization') == 'none' and
        spec.get('label_id') == f'forward_{horizon}_session_open_close_v1' and
        spec.get('start_session_offset') == 1 and spec.get('end_session_offset') == horizon and
        spec.get('start_price') == 'open' and spec.get('end_price') == 'close' and
        spec.get('price_basis') == 'common_anchor_adjusted_v1',
        'matrix evaluation RawLabel endpoint semantics required')
    # Each original adjustment anchor remains in provenance. The target
    # definition and selected actual leaves determine whether keys can share.
    return {k: _copy_wire(v) for k, v in spec.items() if k != 'adjustment_anchor'}


def _target(row, key, calendar, cutoff, horizon, positions=None):
    _require(type(row) is dict and set(row) == TARGET_FIELDS and
        (row['security_id'], row['feature_session']) == key and type(row['valid']) is bool,
        'matrix evaluation Target fields/key mismatch')
    pos = calendar.index(key[1]) if positions is None else positions[key[1]]
    for field, offset in (('start_session', 1), ('end_session', horizon)):
        expected = calendar[pos+offset] if pos+offset < len(calendar) else None
        _require(row[field] == expected, 'matrix evaluation Target calendar endpoint mismatch')
    _require(type(row['source_refs']) is list and bool(row['source_refs']) and
        all(_ref(ref) for ref in row['source_refs']) and
        len(row['source_refs']) == len(set(row['source_refs'])),
        'matrix evaluation Target original source refs required')
    if row['valid']:
        _require(_finite(row['return']) and row['invalid_reason'] is None and
            row['start_session'] is not None and row['end_session'] is not None and
            row['label_available_at'] is not None and _instant(row['label_available_at']) <= cutoff,
            'matrix evaluation valid Target value/clock mismatch')
    else:
        _require(row['return'] is None and type(row['invalid_reason']) is str and
            bool(row['invalid_reason']), 'matrix evaluation invalid Target null/reason required')
        if row['label_available_at'] is not None:
            _instant(row['label_available_at'])


def _merge_evaluation_targets(items, *, calendar, universe, scope):
    """Merge actual leaf identities, retaining every containing slice lineage.

    Each item is (original evaluation slice, per-key actual leaf digest map,
    original small RawLabel provenance map). An actual leaf digest is supplied
    by matrix admission from the selected endpoint/factor versions and clocks;
    it excludes the containing slice and whole query range identities.
    """
    state = _target_state()
    _join_target_slices(items, calendar=calendar, universe=universe, scope=scope, state=state)
    return _finish_targets(state, scope)


def _target_state():
    return {'labels': {}, 'leaves': {}, 'raw': {}, 'slices': {}, 'label_spec': None}


def _join_target_slices(items, *, calendar, universe, scope, state):
    labels, leaves, raw, slices = (state[key] for key in ('labels', 'leaves', 'raw', 'slices'))
    label_spec = state['label_spec']
    positions = {day: index for index, day in enumerate(calendar)}
    wanted = {(security, day) for day in scope['sessions'] for security in scope['universe']}
    for evaluation, bindings, provenance in items:
        _require(isinstance(evaluation, Mapping) and set(evaluation) == SLICE_FIELDS and
            evaluation['contract_version'] == 'stock_matrix_evaluation_target_slice_v1',
            'original matrix evaluation target slice required')
        evaluation = _copy_wire(evaluation)
        _verify_ref(evaluation, 'label_ref')
        _require(_ref(evaluation['prepared_view_ref']) and _ref(evaluation['fold_spec_ref']),
            'matrix evaluation slice original identities required')
        selector = evaluation['selector']
        _require(isinstance(selector, Mapping) and set(selector) == {'path', 'file_digest', 'selector_ref'} and
            _ref(selector['file_digest']) and _ref(selector['selector_ref']),
            'matrix evaluation original selector required')
        cutoff = _instant(evaluation['cutoff'])
        _require(cutoff <= _instant(scope['evaluation_cutoff']), 'matrix evaluation source cutoff exceeds requested cutoff')
        rows = evaluation['rows']
        _require(type(rows) in (list, tuple) and bool(rows), 'matrix evaluation complete Target rows required')
        days = sorted({row['feature_session'] for row in rows})
        _require(set(days) <= set(calendar), 'matrix evaluation Target outside calendar')
        indexed = _grid(rows, universe, days, 'feature_session', 'matrix evaluation Target')
        _require(isinstance(bindings, Mapping) and set(bindings) == set(indexed) and
            all(_ref(ref) for ref in bindings.values()), 'complete per-key matrix evaluation leaf bindings required')
        _require(isinstance(provenance, Mapping) and bool(provenance), 'original matrix RawLabel provenance required')
        source_days = {}
        for ref, header in provenance.items():
            _require(_ref(ref) and isinstance(header, Mapping) and set(header) == RAW_FIELDS and
                header['label_ref'] == ref and header['contract_version'] == 'stock_label_build_v1' and
                _ref(header['source_ref']), 'matrix RawLabel provenance identity mismatch')
            descriptor = header['raw_input']
            _require(isinstance(descriptor, Mapping) and set(descriptor) == {'path', 'file_digest', 'label_ref'} and
                descriptor['label_ref'] == ref and _ref(descriptor['file_digest']),
                'matrix RawLabel original descriptor mismatch')
            _require(header['calendar_ref'] == digest({'contract_version': 'stock_label_calendar_v1',
                'sessions': calendar}), 'matrix RawLabel original calendar mismatch')
            semantics = _label_semantics(header['label_spec'])
            _require(label_spec is None or label_spec == semantics,
                'comparison evaluation Label definition/version conflict')
            label_spec = semantics
            _require(type(header['sessions']) in (list, tuple) and bool(header['sessions']) and
                list(header['sessions']) == sorted(set(header['sessions'])) and set(header['sessions']) <= set(calendar),
                'matrix RawLabel original date coverage required')
            # Copy each small header once, after checking its original identity.
            value = _copy_wire(header)
            _require(ref not in raw or raw[ref] == value, 'conflicting matrix RawLabel provenance')
            raw.setdefault(ref, value)
            for day in header['sessions']:
                source_days.setdefault(day, set()).add(header['source_ref'])
        lineage = {k: deepcopy(v) for k, v in evaluation.items() if k != 'rows'}
        lineage.update(sessions=days, raw_label_refs=sorted(provenance))
        _require(evaluation['label_ref'] not in slices or slices[evaluation['label_ref']] == lineage,
            'conflicting matrix target slice lineage')
        slices.setdefault(evaluation['label_ref'], lineage)
        for key, row in indexed.items():
            _target(row, key, calendar, cutoff, label_spec['horizon_sessions'], positions)
            _require(set(row['source_refs']) <= source_days.get(key[1], set()),
                'matrix evaluation Target source outside original RawLabel provenance')
            if key not in wanted:
                continue
            content = {k: row[k] for k in TARGET_FIELDS if k != 'source_refs'}
            if key in labels:
                old = {k: labels[key][k] for k in TARGET_FIELDS if k != 'source_refs'}
                _require(old == content and leaves[key] == bindings[key],
                    'comparison matrix Label leaf/value/clock conflict')
                labels[key]['source_refs'] = sorted(set(labels[key]['source_refs']) | set(row['source_refs']))
            else:
                labels[key] = deepcopy(row)
                labels[key]['source_refs'] = sorted(row['source_refs'])
                leaves[key] = bindings[key]
    state['label_spec'] = label_spec


def _finish_targets(state, scope):
    labels, leaves, raw, slices = (state[key] for key in ('labels', 'leaves', 'raw', 'slices'))
    wanted = {(security, day) for day in scope['sessions'] for security in scope['universe']}
    _require(set(labels) == wanted, 'complete matrix evaluation Label scope required')
    return {'labels': labels, 'label_leaf_bindings': leaves, 'label_spec': state['label_spec'],
            'raw_provenance': {ref: raw[ref] for ref in sorted(raw)},
            'label_inputs': [slices[ref] for ref in sorted(slices)]}


RAW_METADATA_FIELDS = {'mode', 'label_ref', 'label_spec', 'calendar_ref',
    'snapshot', 'pit_policy', 'sources', 'label_inputs', 'label_shard_refs'}
RECEIPT_FIELDS = {'contract_version', 'signal_inputs', 'raw_label_input', 'scope',
    'source_closure', 'source_records', 'validation_sources', 'receipt_ref',
    'batch_manifest', 'batch_ref'}


def _checked_descriptor(descriptor, name, marks):
    from .stock_signal_evaluation_projection import _read_checked
    _require(type(descriptor) is dict and set(descriptor) == {'path', 'file_digest', name} and
        type(descriptor['path']) is str and Path(descriptor['path']).is_absolute() and
        _ref(descriptor['file_digest']) and _ref(descriptor[name]), 'fixed saved input descriptor required')
    wire, _ = _read_checked(descriptor['path'], descriptor['file_digest'], marks=marks)
    if name == 'label_ref' and wire.get('contract_version') == 'stock_label_bundle_v1':
        _verify_ref(wire, 'label_ref')
        candidates = [wire[key] for key in ('training', 'evaluation') if wire[key]['label_ref'] == descriptor[name]]
        _require(len(candidates) == 1, 'ambiguous/missing saved Raw Label build')
        wire = candidates[0]
    _verify_ref(wire, name)
    _require(wire[name] == descriptor[name], 'saved input content ref mismatch')
    return wire


def _override_leaf_ref(header, row):
    return digest({'raw_label_ref': header['label_ref'], 'file_digest': header['raw_input']['file_digest'],
                   'row': row})


def _label_day_ref(shard):
    return digest({'session': shard['session'], 'rows': [
        {'security_id': row['security_id'], 'label_leaf_ref': row['label_leaf_ref'],
         'label': {key: value for key, value in row['label'].items() if key != 'source_refs'}}
        for row in shard['rows']]})


def _label_projection_ref(raw):
    return digest({key: raw[key] for key in ('label_spec', 'calendar_ref', 'snapshot', 'pit_policy', 'label_shard_refs')})


def _clock_floor_matrix(metadata, raw):
    clocks = [item['evaluation_clock_floor'] for items in metadata.values() for item in items]
    for header in raw['sources'].values():
        context = header['source_context']
        for query in (context['query'], context['derivation']['price_query'], context['derivation']['factor_query']):
            clocks.extend(query['cutoff_by_session'].values())
    return max(map(_instant, clocks)).isoformat().replace('+00:00', 'Z')


def _sources_table(value):
    """Freeze the shared table once; fold leases only borrow its indices."""
    _require(isinstance(value, (list, tuple)), 'shared batch source records required')
    result = {}
    for record in value:
        if type(record) is tuple and len(record) == 2:
            record = {'path': record[0], 'file_digest': record[1]}
        _require(isinstance(record, Mapping) and set(record) == {'path', 'file_digest'} and
            type(record['path']) is str and Path(record['path']).is_absolute() and _ref(record['file_digest']),
            'batch source record fields mismatch')
        _require(record['path'] not in result or result[record['path']] == record['file_digest'],
            'conflicting batch source record')
        result[record['path']] = record['file_digest']
    return result


def _owner_targets(view):
    """Translate the admitted owner's original OOS proof, while its lease lives."""
    evidence = view.label_leaf_bindings
    _require(type(evidence) is dict and set(evidence) == {'rows', 'contents'} and
        type(evidence['rows']) is list and type(evidence['contents']) is dict,
        'matrix owner original Label endpoint proof required')
    bindings, days = {}, {}
    for leaf in evidence['rows']:
        _require(type(leaf) is dict and set(leaf) == {'security_id', 'feature_session',
            'raw_label_ref', 'source_ref', 'query_ref', 'adjustment_anchor',
            'start_open_ref', 'end_close_ref'}, 'matrix owner Label leaf fields mismatch')
        key = leaf['security_id'], leaf['feature_session']
        _require(key not in bindings, 'duplicate matrix owner Label leaf key')
        for field in ('start_open_ref', 'end_close_ref'):
            ref = leaf[field]
            _require(ref is None or (_ref(ref) and ref in evidence['contents'] and
                digest(evidence['contents'][ref]) == ref), 'matrix owner original Label leaf content mismatch')
        # Owner refs identify actual endpoint values and original price/factor
        # provenance. Whole query, Raw build and containing slice refs stay in
        # lineage, so a change of query range alone cannot split an equal leaf.
        bindings[key] = digest({field: leaf[field] for field in ('start_open_ref', 'end_close_ref')})
        days.setdefault(leaf['raw_label_ref'], set()).add(key[1])
    _require(type(view.raw_provenance) is list and bool(view.raw_provenance),
        'matrix owner original RawLabel headers required')
    provenance = {}
    for raw in view.raw_provenance:
        _require(type(raw) is dict and set(raw) == {'raw_build', 'contract_version',
            'label_ref', 'label_spec', 'calendar_ref', 'source_ref', 'source_evidence'} and
            raw['label_ref'] in days, 'matrix owner original RawLabel fields/coverage mismatch')
        header = {key: _copy_wire(raw[key]) for key in
            ('contract_version', 'label_ref', 'label_spec', 'calendar_ref', 'source_ref')}
        header.update(raw_input=_copy_wire(raw['raw_build']),
            source_context=_copy_wire(raw['source_evidence']['context']), sessions=sorted(days[raw['label_ref']]))
        _require(raw['label_ref'] not in provenance, 'duplicate matrix owner RawLabel identity')
        provenance[raw['label_ref']] = header
    for leaf in evidence['rows']:
        header = provenance[leaf['raw_label_ref']]
        _require(leaf['source_ref'] == header['source_ref'] and
            leaf['query_ref'] == digest(header['source_context']['query']) and
            leaf['adjustment_anchor'] == header['label_spec']['adjustment_anchor'],
            'matrix owner Label leaf original lineage mismatch')
    return view.evaluation, bindings, provenance


def _compact_features(indexed):
    """Retain only the already checked membership and frozen clock/ref fields."""
    members, predictions = {}, {}
    for key, row in indexed.items():
        members[key] = {'member': row['member']}
        available = max((_instant(clock) for clock in row['availability'] if clock is not None), default=None)
        predictions[key] = {'knowledge_cutoff': row['knowledge_cutoff'],
            'feature_available_at': available.isoformat().replace('+00:00', 'Z') if available is not None else None,
            'source_refs': deepcopy(row['source_refs'])}
    return members, predictions


def _admit_matrix(signal_inputs, raw_label_input, scope, batch):
    from .stock_batch import _data
    from .stock_fold_artifacts import load_stock_ml_fold
    from .stock_signal_evaluation_projection import _read_checked, _check_marks, _mark
    _data(batch)
    manifest = batch.to_dict()
    _require(manifest['contract_version'] == 'stock_ml_batch_inputs_v2', 'verified matrix batch required')
    _require(callable(getattr(batch, '_project_evaluation', None)), 'matrix owner OOS evaluation interface required')
    batch._check_sources()
    _require(type(signal_inputs) is dict and bool(signal_inputs) and
        all(type(name) is str and name for name in signal_inputs), 'ordered Signal mapping required')
    projected, metadata, refs, closures = {}, {}, {}, {}
    target_state = _target_state()
    shared_table = batch._evaluation_source_records()
    _require(type(shared_table) is tuple and all(type(record) is tuple for record in shared_table),
        'matrix owner immutable shared source table required')
    records, marks = _sources_table(shared_table), {}
    marks.update({Path(p): _mark(Path(p).stat()) for p in records})
    common = None
    for name, descriptors in signal_inputs.items():
        descriptors = descriptors if type(descriptors) is list else [descriptors]
        _require(bool(descriptors), 'nonempty ordered matrix Signal list required')
        rows, members, prediction_features = {}, {}, {}
        metadata[name], refs[name], closures[name] = [], [], []
        previous = None
        for descriptor in descriptors:
            signal = _checked_descriptor(descriptor, 'signal_run_ref', marks)
            path = Path(descriptor['path']).parent
            _require(Path(descriptor['path']).name == 'predictions.json' and
                signal['contract_version'] == 'stock_prediction_run_v2', 'saved matrix predictions.json required')
            # The public owner loader retains every saved output check. Its
            # read-only matrix path must use the OOS lease, never training X/y/P.
            fold_manifest, manifest_digest = _read_checked(path/'manifest.json', marks=marks)
            own_paths = {str(path/'manifest.json'), *(str(path/filename) for filename in fold_manifest['files'])}
            for source_path in own_paths:
                marks.setdefault(Path(source_path), _mark(Path(source_path).stat()))
            _require(fold_manifest['files']['predictions.json'] == descriptor['file_digest'],
                'matrix Signal descriptor/manifest bytes mismatch')
            load_stock_ml_fold(path, batch=batch)
            fold, _ = _read_checked(path/'fold.json', fold_manifest['files']['fold.json'], marks=marks)
            model, _ = _read_checked(path/'model.json', fold_manifest['files']['model.json'], marks=marks)
            features, _ = _read_checked(path/'feature-slice.json', fold_manifest['files']['feature-slice.json'], marks=marks)
            _require(fold['contract_version'] == 'stock_ml_fold_v3' and
                fold['signal_run_ref'] == signal['signal_run_ref'] and
                model['model_ref'] == signal['model_ref'], 'original matrix fold/Signal binding mismatch')
            inputs, spec = fold['definition']['input_manifest'], fold['definition']['fold_spec']
            with batch._project_evaluation(inputs, spec) as view:
                definition = _copy_wire(view.common)
                identity = {key: definition[key] for key in ('snapshot', 'pit_policy', 'calendar', 'universe')}
                _require(common is None or common == identity, 'comparison matrix common scope mismatch')
                common = identity
                _require(common['calendar'] == scope['calendar'] and
                    set(scope['universe']) <= set(common['universe']), 'matrix evaluation frozen axes mismatch')
                _require(features == _copy_wire(view.features), 'saved OOS Feature slice mismatch')
                indices = tuple(view.source_record_indices)
                _require(all(type(index) is int and 0 <= index < len(shared_table) for index in indices),
                    'matrix consumed source indices mismatch')
                consumed = {shared_table[index][0] for index in indices}
                if raw_label_input is None:
                    # Copy only this fold's small original proof and OOS rows.
                    _join_target_slices([_owner_targets(view)],
                        calendar=common['calendar'], universe=common['universe'], scope=scope, state=target_state)
            days = features['prediction_sessions']; universe = common['universe']
            feature_index = _grid(features['rows'], universe, days, 'session', 'OOS Feature')
            indexed = _grid(signal['rows'], universe, days, 'session', 'matrix Signal')
            _require(previous is None or previous < days[0], 'weekly Signals must be ordered and disjoint')
            previous = days[-1]
            _require(not rows.keys() & indexed.keys(), 'duplicate weekly Signal key')
            _require(signal['signal_stage'] == 'prediction_raw' and signal['score_unit'] == 'dimensionless' and
                signal['score_semantics'] == model['target_semantics'] and signal['feature_ref'] == features['feature_ref'] and
                _instant(model['fit_cutoff']) <= _instant(scope['evaluation_cutoff']), 'matrix Signal stage/model clock mismatch')
            clocks = [model['fit_cutoff'], model['simulated_available_at']]
            for key, prediction in indexed.items():
                original = feature_index[key]
                _require(prediction['member'] is original['member'] and type(prediction['member']) is bool and
                    type(prediction['valid']) is bool, 'matrix Signal original member mismatch')
                knowledge, available = _instant(prediction['knowledge_cutoff']), _instant(prediction['available_at'])
                _require(available <= knowledge <= _instant(scope['evaluation_cutoff']) and
                    _instant(original['knowledge_cutoff']) <= knowledge and
                    all(clock is None or _instant(clock) <= _instant(original['knowledge_cutoff']) for clock in original['availability']),
                    'matrix OOS source clock conflict')
                clocks.extend([prediction['knowledge_cutoff'], original['knowledge_cutoff']])
                if prediction['valid']:
                    _require(original['member'] and all(original['validity']) and all(_finite(v) for v in original['values']) and
                        _finite(prediction['score']) and prediction['invalid_reason'] is None, 'matrix valid Signal/Feature mismatch')
                else:
                    _require(prediction['score'] is None and bool(prediction['invalid_reason']), 'matrix invalid Signal null/reason required')
            fold_members, fold_predictions = _compact_features(feature_index)
            for key, member in fold_members.items():
                _require(key not in members or members[key]['member'] is member['member'], 'weekly historical membership conflict')
                members[key] = member
            rows.update(indexed); prediction_features.update(fold_predictions)
            for filename, file_ref in fold_manifest['files'].items():
                source_path = str(path/filename)
                _require(source_path not in records or records[source_path] == file_ref, 'conflicting matrix fold output pin')
                records[source_path] = file_ref
            records[str(path/'manifest.json')] = manifest_digest
            closures[name].append({'signal_input': deepcopy(descriptor), 'model_ref': model['model_ref'],
                'feature_ref': features['feature_ref'], 'source_paths': sorted(consumed | own_paths)})
            refs[name].append(signal['signal_run_ref'])
            metadata[name].append({'signal_contract_version': signal['contract_version'],
                'signal_run_ref': signal['signal_run_ref'], 'prediction_sessions': list(days),
                'signal_stage': signal['signal_stage'], 'score_unit': signal['score_unit'],
                'score_semantics': signal['score_semantics'], 'model': model,
                'feature_contract_version': features['contract_version'], 'feature_ref': features['feature_ref'],
                'snapshot': common['snapshot'], 'pit_policy': common['pit_policy'],
                'evaluation_clock_floor': max(map(_instant, clocks)).isoformat().replace('+00:00', 'Z')})
            # The next saved-fold admission must have no reference to this
            # fold's wide rows, including the last loop row and closed lease.
            del features, feature_index, original, view, fold_members, fold_predictions
        projected[name] = {'rows': rows, 'members': members, 'prediction_features': prediction_features}
    if raw_label_input is not None:
        raw_wire = _checked_descriptor(raw_label_input, 'label_ref', marks)
        original, labels = _validate_raw_labels(raw_wire, scope, common['snapshot'], common['pit_policy'])
        from .stock_signal_evaluation_projection import _raw_metadata
        header = {**_raw_metadata(original), 'raw_input': deepcopy(raw_label_input),
            'sessions': sorted({row['feature_session'] for row in original['rows']})}
        keys = [(security, day) for day in scope['sessions'] for security in scope['universe']]
        targets = {'labels': {key: labels[key] for key in keys},
            'label_leaf_bindings': {key: _override_leaf_ref(header, labels[key]) for key in keys},
            'raw_provenance': {original['label_ref']: header}, 'label_inputs': [], 'label_spec': deepcopy(original['label_spec'])}
        label_ref, mode = original['label_ref'], 'raw_override'
        source_path = raw_label_input['path']
        _require(source_path not in records or records[source_path] == raw_label_input['file_digest'], 'conflicting shared evaluation Label pin')
        records[source_path] = raw_label_input['file_digest']
    else:
        targets = _finish_targets(target_state, scope)
        label_ref, mode = None, 'fold_targets'
    raw = {'mode': mode, 'label_ref': label_ref, 'label_spec': targets['label_spec'],
        'calendar_ref': digest({'contract_version': 'stock_label_calendar_v1', 'sessions': scope['calendar']}),
        'snapshot': common['snapshot'], 'pit_policy': common['pit_policy'],
        'sources': targets['raw_provenance'], 'label_inputs': targets['label_inputs'], 'label_shard_refs': {}}
    admitted = {'projected': projected, 'closures': closures, 'refs': refs, 'metadata': metadata,
        'raw': raw, 'labels': targets['labels'], 'label_leaf_bindings': targets['label_leaf_bindings'], 'scope': scope}
    from .stock_signal_evaluation_projection import _shard
    for day in scope['sessions']:
        raw['label_shard_refs'][day] = _label_day_ref(_shard(admitted, scope, day))
    if mode == 'fold_targets': raw['label_ref'] = _label_projection_ref(raw)
    _select_inputs(admitted, scope)
    batch._check_sources(); _check_marks(marks)
    return admitted, records, marks, manifest


def _save_matrix_inputs(signal_inputs, raw_label_input, scope, destination, batch):
    from .contracts import ArtifactRef
    from .stock_signal_evaluation_projection import (
        MATRIX_INPUT_VERSION, _root_id, _shard, _load_inputs, _check_marks)
    admitted, records, marks, manifest = _admit_matrix(signal_inputs, raw_label_input, scope, batch)
    table = [{'path': path, 'file_digest': records[path]} for path in sorted(records)]
    offsets = {record['path']: index for index, record in enumerate(table)}
    closure = {name: [{**{key: value for key, value in source.items() if key != 'source_paths'},
        'source_record_indices': [offsets[path] for path in source['source_paths']]}
        for source in sources] for name, sources in admitted['closures'].items()}
    names = ('stock_signal_evaluation_matrix.py', 'stock_signal_evaluation_projection.py',
        'stock_signal_evaluation_inputs.py', 'stock_batch.py', 'stock_matrix_reader.py',
        'stock_matrix_folds.py', 'stock_fold_artifacts.py', 'stock_artifacts.py', 'stock_label_contracts.py')
    receipt = {'contract_version': 'stock_signal_evaluation_admission_v2',
        'signal_inputs': deepcopy(signal_inputs), 'raw_label_input': deepcopy(raw_label_input), 'scope': scope,
        'batch_manifest': manifest, 'batch_ref': manifest['batch_ref'], 'source_records': table,
        'source_closure': closure, 'validation_sources': {
            name: file_digest(Path(__file__).parent/name) for name in names}}
    receipt['receipt_ref'] = digest(receipt)
    root = {'contract_version': MATRIX_INPUT_VERSION, 'scope': scope,
        'signal_order': list(signal_inputs), 'signal_refs': admitted['refs'], 'signal_metadata': admitted['metadata'],
        'raw_metadata': admitted['raw'], 'clock_floor': _clock_floor_matrix(admitted['metadata'], admitted['raw']),
        'admission_receipt': receipt, 'shards': {}}
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.signal-inputs-', dir=destination) as temporary:
        stage = Path(temporary)/'complete'; stage.mkdir()
        for day in scope['sessions']:
            name = day+'.json'; write_json(stage/name, _shard(admitted, scope, day))
            root['shards'][day] = {'file': name, 'file_digest': file_digest(stage/name)}
        root['input_id'] = _root_id(root); write_json(stage/'manifest.json', root)
        ref = ArtifactRef(artifact_type='StockSignalEvaluationInputs', artifact_id=root['input_id'],
            artifact_contract_version=MATRIX_INPUT_VERSION, content_digest=file_digest(stage/'manifest.json'),
            uri=str(stage/'manifest.json'))
        # The caller owns the batch backing; release this full OOS projection
        # before reading the frozen rows again. No training matrix is retained.
        del admitted
        _load_inputs(ref, scope)
        batch._check_sources(); _check_marks(marks)
        target = destination/root['input_id'][7:]
        final = ArtifactRef(artifact_type=ref.artifact_type, artifact_id=ref.artifact_id,
            artifact_contract_version=ref.artifact_contract_version, content_digest=ref.content_digest,
            uri=str(target/'manifest.json'))
        if target.exists():
            _load_inputs(final, scope); batch._check_sources(); _check_marks(marks)
            return final
        try:
            os.rename(stage, target)
        except OSError:
            if not target.exists(): raise
            _load_inputs(final, scope)
        return final


def _verify_matrix_root(root, ref, scope):
    from .stock_signal_evaluation_projection import MATRIX_INPUT_VERSION, ROOT_FIELDS, _root_id, _expand_closure
    from .stock_signal_evaluation_inputs import _scope
    _require(type(root) is dict and set(root) == ROOT_FIELDS and root['contract_version'] == MATRIX_INPUT_VERSION and
        root['input_id'] == ref.artifact_id == _root_id(root), 'frozen matrix input identity/fields mismatch')
    base = _scope(root['scope'])
    _require(base == root['scope'] and scope['calendar'] == base['calendar'] and
        set(scope['sessions']) <= set(base['sessions']) and set(scope['universe']) <= set(base['universe']),
        'evaluation outside frozen input scope/calendar')
    names = root['signal_order']
    _require(type(names) is list and bool(names) and all(type(name) is str and name for name in names) and
        len(names) == len(set(names)) and set(names) == set(root['signal_refs']) == set(root['signal_metadata']),
        'frozen comparison axes mismatch')
    receipt = root['admission_receipt']; _verify_ref(receipt, 'receipt_ref')
    _require(set(receipt) == RECEIPT_FIELDS and receipt['contract_version'] == 'stock_signal_evaluation_admission_v2' and
        receipt['scope'] == base and set(receipt['signal_inputs']) == set(names) == set(receipt['source_closure']),
        'frozen matrix admission receipt mismatch')
    manifest = receipt['batch_manifest']; _verify_ref(manifest, 'content_digest')
    _require(manifest['contract_version'] == 'stock_ml_batch_inputs_v2' and manifest['status'] == 'COMPLETE' and
        manifest['batch_ref'] == receipt['batch_ref'] == digest({key: value for key, value in manifest.items()
            if key not in ('content_digest', 'batch_ref')}), 'frozen matrix batch identity mismatch')
    raw = root['raw_metadata']
    _require(type(raw) is dict and set(raw) == RAW_METADATA_FIELDS and raw['mode'] in ('fold_targets', 'raw_override') and
        raw['calendar_ref'] == digest({'contract_version': 'stock_label_calendar_v1', 'sessions': base['calendar']}) and
        type(raw['snapshot']) is str and raw['snapshot'] not in ('', 'latest', 'current') and
        type(raw['pit_policy']) is str and bool(raw['pit_policy']) and
        type(raw['sources']) is dict and bool(raw['sources']) and
        set(raw['label_shard_refs']) == set(base['sessions']) and all(_ref(x) for x in raw['label_shard_refs'].values()),
        'frozen matrix Label metadata mismatch')
    records = _sources_table(receipt['source_records'])
    for raw_ref, header in raw['sources'].items():
        _require(type(header) is dict and set(header) == RAW_FIELDS and header['label_ref'] == raw_ref and
            _ref(raw_ref) and type(header['sessions']) is list and bool(header['sessions']) and
            header['sessions'] == sorted(set(header['sessions'])) and set(header['sessions']) <= set(base['calendar']),
            'frozen original RawLabel header/date mismatch')
        descriptor = header['raw_input']
        _require(type(descriptor) is dict and set(descriptor) == {'path', 'file_digest', 'label_ref'} and
            descriptor['label_ref'] == raw_ref and records.get(descriptor['path']) == descriptor['file_digest'],
            'frozen RawLabel original byte binding mismatch')
        wire = {key: header[key] for key in ('contract_version', 'label_ref', 'label_spec', 'calendar_ref', 'source_ref')}
        wire['source_evidence'] = {'context': header['source_context']}
        _validate_raw_label_header(wire, scope, raw['snapshot'], raw['pit_policy'])
        expected_spec = header['label_spec'] if raw['mode'] == 'raw_override' else _label_semantics(header['label_spec'])
        _require(raw['label_spec'] == expected_spec, 'frozen comparison evaluation Label definition/version conflict')
    if raw['mode'] == 'raw_override':
        _require(len(raw['sources']) == 1 and raw['label_inputs'] == [] and receipt['raw_label_input'] is not None,
            'frozen common RawLabel override mismatch')
        header = next(iter(raw['sources'].values()))
        _require(header['raw_input'] == receipt['raw_label_input'] and raw['label_ref'] == header['label_ref'],
            'frozen shared evaluation Label ref mismatch')
    else:
        _require(receipt['raw_label_input'] is None and raw['label_ref'] == _label_projection_ref(raw) and
            type(raw['label_inputs']) is list and bool(raw['label_inputs']), 'frozen default target projection mismatch')
        target_refs = []
        for target in raw['label_inputs']:
            _require(set(target) == (SLICE_FIELDS-{'rows'}) | {'sessions', 'raw_label_refs'} and
                target['contract_version'] == 'stock_matrix_evaluation_target_slice_v1' and
                _ref(target['label_ref']) and _ref(target['fold_spec_ref']) and _ref(target['prepared_view_ref']) and
                target['sessions'] == sorted(set(target['sessions'])) and
                set(target['raw_label_refs']) <= set(raw['sources']), 'frozen original target slice lineage mismatch')
            target_refs.append(target['label_ref'])
        _require(target_refs == sorted(set(target_refs)), 'frozen target slice order/identity mismatch')
    expanded = _expand_closure(receipt)
    by_day = {name: {} for name in names}
    for name in names:
        metadata, sources = root['signal_metadata'][name], expanded[name]
        descriptors = receipt['signal_inputs'][name]
        descriptors = descriptors if type(descriptors) is list else [descriptors]
        _require(type(metadata) is list and bool(metadata) and len(metadata) == len(sources) == len(descriptors) and
            [item['signal_run_ref'] for item in metadata] == root['signal_refs'][name] ==
            [source['signal_input']['signal_run_ref'] for source in sources] and
            [source['signal_input'] for source in sources] == descriptors, 'frozen matrix Signal version binding mismatch')
        for item, source in zip(metadata, sources):
            descriptor = source['signal_input']
            _require(records.get(descriptor['path']) == descriptor['file_digest'],
                'frozen Signal original byte binding mismatch')
            _verify_ref(item['model'], 'model_ref')
            _require(item['signal_contract_version'] == 'stock_prediction_run_v2' and
                item['feature_contract_version'] == 'stock_feature_slice_v3' and
                item['model']['model_ref'] == source['model_ref'] and item['feature_ref'] == source['feature_ref'] and
                item['signal_stage'] == 'prediction_raw' and item['score_unit'] == 'dimensionless' and
                item['score_semantics'] == item['model']['target_semantics'] and
                (item['snapshot'], item['pit_policy']) == (raw['snapshot'], raw['pit_policy']),
                'frozen matrix Signal stage/model/Snapshot/PIT mismatch')
            days = item['prediction_sessions']
            _require(type(days) is list and bool(days) and days == sorted(set(days)) and set(days) <= set(base['calendar']),
                'frozen matrix ordered prediction dates required')
            for day in days:
                _require(day not in by_day[name], 'frozen weekly Signal date overlap')
                by_day[name][day] = item
    _require(root['clock_floor'] == _clock_floor_matrix(root['signal_metadata'], raw) and
        _instant(scope['evaluation_cutoff']) >= _instant(root['clock_floor']),
        'evaluation cutoff precedes frozen source revision/Signal visibility')
    _require(set(root['shards']) == set(base['sessions']) and all(
        set(descriptor) == {'file', 'file_digest'} and descriptor['file'] == day+'.json' and _ref(descriptor['file_digest'])
        for day, descriptor in root['shards'].items()), 'complete frozen matrix date shards required')
    return by_day


def _matrix_label_context(root):
    sources = {day: {} for day in root['scope']['sessions']}
    for header in root['raw_metadata']['sources'].values():
        compiled = (header, _instant(header['source_context']['derivation']['decision_cutoff']),
                    set(header['source_context']['query']['sessions']))
        for day in header['sessions']:
            if day in sources: sources[day][header['source_ref']] = compiled
    return {'sources': sources, 'positions': {day: index for index, day in enumerate(root['scope']['calendar'])}}


def _validate_matrix_label(root, row, key, context):
    raw, label = root['raw_metadata'], row['label']
    _require(_ref(row['label_leaf_ref']), 'frozen matrix actual Label leaf identity required')
    available_sources = context['sources'][key[1]]
    _require(type(label) is dict and set(label) == TARGET_FIELDS and
        type(label['source_refs']) is list and bool(label['source_refs']) and
        label['source_refs'] == sorted(set(label['source_refs'])) and
        set(label['source_refs']) <= set(available_sources), 'frozen matrix Label original source binding mismatch')
    if raw['mode'] == 'raw_override':
        header = next(iter(raw['sources'].values()))
        _require(label['source_refs'] == [header['source_ref']] and
            row['label_leaf_ref'] == _override_leaf_ref(header, label), 'frozen common Label original leaf/byte mismatch')
    for source_ref in label['source_refs']:
        header, native_cutoff, query_sessions = available_sources[source_ref]
        _target(label, key, root['scope']['calendar'], native_cutoff,
                raw['label_spec']['horizon_sessions'], context['positions'])
        if label['valid']:
            _require(label['start_session'] in query_sessions and label['end_session'] in query_sessions,
                'frozen valid Label endpoints outside original query')


def _audit_matrix_input(ref):
    from .stock_batch import load_stock_ml_batch_inputs
    from .stock_signal_evaluation_projection import _load_inputs, _read_checked, _check_marks, _shard
    marks = {}
    root, _ = _read_checked(ref.uri, ref.content_digest)
    _, root, _ = _load_inputs(ref, root['scope'], marks=marks)
    receipt = root['admission_receipt']
    # A fresh public batch admission hashes its common closure once. Do not
    # separately hash that same large graph before calling the owner loader.
    with load_stock_ml_batch_inputs(receipt['batch_manifest']) as batch:
        original, records, source_marks, _ = _admit_matrix(
            {name: receipt['signal_inputs'][name] for name in root['signal_order']},
            receipt['raw_label_input'], root['scope'], batch)
        offsets = {record['path']: index for index, record in enumerate(receipt['source_records'])}
        closure = {name: [{**{key: value for key, value in source.items() if key != 'source_paths'},
            'source_record_indices': [offsets[path] for path in source['source_paths']]}
            for source in sources] for name, sources in original['closures'].items()}
        _require(records == _sources_table(receipt['source_records']) and closure == receipt['source_closure'] and
            original['metadata'] == root['signal_metadata'] and
            original['raw'] == root['raw_metadata'], 'audit original matrix source/metadata mismatch')
        for day, descriptor in root['shards'].items():
            shard, _ = _read_checked(Path(ref.uri).parent/descriptor['file'], descriptor['file_digest'], marks=marks)
            _require(_shard(original, root['scope'], day) == shard, 'audit original matrix date projection mismatch')
        batch._check_sources(); _check_marks(source_marks)
    _check_marks(marks)
    return ref

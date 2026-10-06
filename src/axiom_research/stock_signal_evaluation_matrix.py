"""Keyed OOS evaluation admission for already verified matrix batches.

The matrix owner supplies the OOS lease and saved-fold admission. This module
only joins those original targets; all statistics remain in the existing Core.
"""
from collections.abc import Mapping
from copy import deepcopy

from .stock_artifacts import digest, _verify_ref
from .stock_label_contracts import _instant, _finite
from .stock_signal_evaluation_inputs import _grid, _require


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


def _target(row, key, calendar, cutoff, horizon):
    _require(type(row) is dict and set(row) == TARGET_FIELDS and
        (row['security_id'], row['feature_session']) == key and type(row['valid']) is bool,
        'matrix evaluation Target fields/key mismatch')
    pos = calendar.index(key[1])
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
    labels, leaves, raw, slices = {}, {}, {}, {}
    label_spec = None
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
            _target(row, key, calendar, cutoff, label_spec['horizon_sessions'])
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
    _require(set(labels) == wanted, 'complete matrix evaluation Label scope required')
    return {'labels': labels, 'label_leaf_bindings': leaves, 'label_spec': label_spec,
            'raw_provenance': {ref: raw[ref] for ref in sorted(raw)},
            'label_inputs': [slices[ref] for ref in sorted(slices)]}

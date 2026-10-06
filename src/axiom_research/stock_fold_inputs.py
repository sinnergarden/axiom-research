"""Validate explicit saved parents and select a fold; no numerical executor.

Common proofs are read and released one at a time. Validated Feature rows can
be reused by a process-local batch; this is not a streaming proof loader or a
general storage/path migration service.
"""
from copy import deepcopy
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from weakref import WeakKeyDictionary

from .stock_artifacts import (digest, file_digest, _read, _verify_ref,
                              _canonical_file_ref, _verify_normalized_labels)
from .stock_label_contracts import (_instant, _session, _finite,
                                    NORMALIZATION_SPEC, normalization_section_inputs)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seal(value, key):
    return {**value, key: digest(value)}


def file_fingerprint(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def ordered(values, name):
    require(type(values) is list and bool(values) and
            all(type(v) is str and v for v in values) and values == sorted(set(values)),
            'ordered unique ' + name + ' required')
    return values


def read_parent(descriptor, ref_key):
    require(set(descriptor) == {'path', 'file_digest', ref_key}, 'explicit parent descriptor required')
    require(type(descriptor['path']) is str and Path(descriptor['path']).is_absolute(), 'fixed absolute parent path required')
    require(file_digest(descriptor['path']) == descriptor['file_digest'], 'parent file digest mismatch')
    value = _read(descriptor['path'])
    _verify_ref(value, ref_key)
    require(value[ref_key] == descriptor[ref_key], 'parent content ref mismatch')
    require(file_digest(descriptor['path']) == descriptor['file_digest'], 'parent changed during read')
    return value


def grid(rows, dates, universe, date_key):
    expected = {(s, d) for d in dates for s in universe}
    result = {}
    for row in rows:
        key = row.get('security_id'), row.get(date_key)
        require(key in expected and key not in result, 'duplicate or unexpected saved grid key')
        result[key] = row
    require(set(result) == expected, 'complete saved security/session grid required')
    return result


def feature_available(row):
    values = row['availability']
    return max((v for v in values if v is not None), key=_instant, default=None)


_COMMON_SOURCE_KEYS = ('scope', 'snapshot', 'pit_policy', 'calendar', 'universe',
    'catalog_ref', 'feature_selection', 'ordered_features', 'feature_parents')
_FEATURE_INPUT_TOKEN = object()
_FEATURE_INPUT_DATA = WeakKeyDictionary()


class _CopyingMapping(Mapping):
    """Expose indexed saved values without lending their mutable contents."""

    __slots__ = ('__values',)

    def __init__(self, values):
        self.__values = values

    def __getitem__(self, key):
        return deepcopy(self.__values[key])

    def __iter__(self):
        return iter(self.__values)

    def __len__(self):
        return len(self.__values)


def _feature_input_data(value):
    require(type(value) is _SavedFeatureInputs and value in _FEATURE_INPUT_DATA,
            'validated saved Feature inputs required')
    return _FEATURE_INPUT_DATA[value]


class _SavedFeatureInputs:
    """Process-local validated parents, constructed only by the saved loader.

    Public mappings and builds return copies. Private borrowing is reserved for
    batch orchestration and projection, which must leave the saved values alone.
    The registry also rejects uninitialized or copied instances.
    """

    __slots__ = ('__weakref__',)

    def __init__(self, token, *, identity, rows, parents, builds):
        require(token is _FEATURE_INPUT_TOKEN, 'saved Feature inputs require validation')
        _FEATURE_INPUT_DATA[self] = {'identity': identity, 'rows': rows,
            'parents': parents, 'builds': builds}

    @property
    def common_source_identity(self):
        return _feature_input_data(self)['identity']

    @property
    def rows(self):
        return _CopyingMapping(self._rows)

    @property
    def parents_by_session(self):
        return _CopyingMapping(self._parents_by_session)

    @property
    def feature_builds(self):
        return _CopyingMapping(_feature_input_data(self)['builds'])

    def feature_build(self, feature_ref):
        return deepcopy(self._borrow_feature_build(feature_ref))

    @property
    def _rows(self):
        return _feature_input_data(self)['rows']

    @property
    def _parents_by_session(self):
        return _feature_input_data(self)['parents']

    def _borrow_feature_build(self, feature_ref):
        return _feature_input_data(self)['builds'][feature_ref]


def common_source_identity(manifest):
    """Bind only the immutable source definition shared by different fits."""
    require(type(manifest) is dict and all(key in manifest for key in _COMMON_SOURCE_KEYS),
            'saved common source definition required')
    return digest({key: manifest[key] for key in _COMMON_SOURCE_KEYS})


def _validate_saved_manifest(manifest):
    require(set(manifest) == {'contract_version', 'scope', 'snapshot', 'pit_policy', 'calendar',
        'universe', 'catalog_ref', 'feature_selection', 'ordered_features', 'feature_parents',
        'training_labels', 'evaluation_labels'} and
        manifest['contract_version'] == 'stock_ml_saved_inputs_v1', 'unsupported saved-input manifest')
    require(type(manifest['snapshot']) is str and manifest['snapshot'] not in ('', 'current', 'latest') and
            type(manifest['pit_policy']) is str and bool(manifest['pit_policy']), 'fixed Snapshot/PIT required')
    calendar = ordered(manifest['calendar'], 'calendar'); [_session(d) for d in calendar]
    universe = ordered(manifest['universe'], 'universe')
    columns = manifest['ordered_features']
    require(type(columns) is list and bool(columns) and len(set(columns)) == len(columns), 'ordered columns required')
    return calendar, universe, columns


def validate_spec(spec, calendar):
    require(set(spec) == {'contract_version', 'training_window', 'fit_session', 'fit_cutoff',
        'simulated_model_available_at', 'oos_trade_sessions', 'inference_cutoff_by_session',
        'evaluation_cutoff'} and spec['contract_version'] in
        ('stock_ml_fold_spec_v1', 'stock_ml_fold_spec_v2'), 'unsupported fold spec')
    window = spec['training_window']
    fit = spec['fit_session']; require(fit in calendar, 'fit session outside frozen calendar')
    i = calendar.index(fit)
    if spec['contract_version'] == 'stock_ml_fold_spec_v1':
        require(window == {'unit': 'feature_sessions', 'length': 65, 'end': 'previous_fit_session'} and
                type(window['length']) is int, 'this bounded profile requires 65 actual feature sessions')
        require(i >= 65, 'insufficient frozen training calendar')
        training = calendar[i-65:i]
    else:
        require(window == {'unit': 'calendar_years', 'length': 2, 'end': 'previous_fit_session',
                'start': 'fit_date_minus_years_inclusive', 'leap_day': 'clamp_feb_28'} and
                type(window['length']) is int, 'this bounded profile requires two calendar years')
        ordered(calendar, 'calendar'); [_session(d) for d in calendar]
        fit_date = date.fromisoformat(fit)
        try:
            boundary = fit_date.replace(year=fit_date.year-2).isoformat()
        except ValueError:
            require(fit_date.month == 2 and fit_date.day == 29, 'unsupported calendar-year boundary')
            boundary = fit_date.replace(year=fit_date.year-2, day=28).isoformat()
        require(calendar[0] < boundary, 'insufficient frozen two-year training calendar/lookback')
        training = [day for day in calendar[:i] if day >= boundary]
        require(bool(training) and training[-1] == calendar[i-1],
                'missing previous-fit training session')
    trades = ordered(spec['oos_trade_sessions'], 'OOS trade sessions')
    require(all(d in calendar and calendar.index(d) > 0 for d in trades), 'OOS outside frozen calendar')
    prediction = [calendar[calendar.index(d)-1] for d in trades]
    clocks = spec['inference_cutoff_by_session']
    require(set(clocks) == set(prediction), 'strict previous-session inference clocks required')
    fit_time = _instant(spec['fit_cutoff']); available = _instant(spec['simulated_model_available_at'])
    require(fit_time == _instant(fit+'T20:30:00+08:00') and
            available == _instant(fit+'T20:45:00+08:00'), 'bounded fit/model clock conflict')
    require(all(_instant(clocks[d]) == _instant(d+'T21:00:00+08:00') and
                fit_time < available < _instant(clocks[d]) for d in prediction), 'fold inference clock conflict')
    require(_instant(spec['evaluation_cutoff']) > max(map(_instant, clocks.values())),
            'OOS evaluation cutoff must follow inference')
    return training, prediction


def validate_raw(raw, manifest, cutoff):
    require(raw['contract_version'] == 'stock_label_build_v1' and
            raw['calendar_ref'] == digest({'contract_version': 'stock_label_calendar_v1',
                'sessions': manifest['calendar']}), 'raw label calendar/contract mismatch')
    spec = raw['label_spec']
    require(spec['label_id'] == 'forward_5_session_open_close_v1' and
            spec['horizon_sessions'] == 5 and spec['normalization'] == 'none', 'raw label semantics mismatch')
    ctx = raw['source_evidence']['context']; query = ctx['query']
    require(ctx['snapshot_id'] == manifest['snapshot'] and query['pit_policy'] == manifest['pit_policy'] and
            query['purpose'] == 'label_outcomes' and set(query['symbols']) == set(manifest['universe']) and
            all(_instant(c) <= _instant(cutoff) for c in query['cutoff_by_session'].values()),
            'raw label frozen source/cutoff mismatch')
    dates = sorted({r['feature_session'] for r in raw['rows']})
    require(set(dates) <= set(manifest['calendar']), 'raw dates outside frozen calendar')
    indexed = grid(raw['rows'], dates, manifest['universe'], 'feature_session')
    for (security, day), row in indexed.items():
        i = manifest['calendar'].index(day)
        for field, offset in (('start_session', 1), ('end_session', 5)):
            endpoint = manifest['calendar'][i+offset] if i+offset < len(manifest['calendar']) else None
            require(row.get(field) is None or row[field] == endpoint, 'raw label endpoint conflict')
        if row['valid']:
            require(_finite(row['return']) and row.get('end_session') is not None and
                    _instant(row['label_available_at']) <= _instant(cutoff), 'invalid mature raw value/clock')
    return indexed



def validate_feature_parent(desc, manifest):
    """The original v1 per-parent byte/ref/key/clock validator, without accumulation."""
    calendar, universe, columns = manifest['calendar'], manifest['universe'], manifest['ordered_features']
    require(set(desc) == {'features', 'input_evidence', 'sessions'}, 'Feature parent descriptor required')
    dates = ordered(desc['sessions'], 'Feature parent dates')
    feature = read_parent(desc['features'], 'feature_ref'); parent_ref = feature['feature_ref']
    require(feature['contract_version'] == 'stock_feature_build_v1' and
            feature['catalog_ref'] == manifest['catalog_ref'] and
            feature['selection'] == manifest['feature_selection'] and
            feature['ordered_features'] == columns and
            feature['qlib_view']['snapshot_id'] == manifest['snapshot'], 'Feature parent schema/source mismatch')
    queries = feature['qlib_view']['queries'] + [feature['qlib_view']['universe_query']]
    require(bool(feature['qlib_view']['queries']) and all(isinstance(q, dict) and
            q['pit_policy'] == manifest['pit_policy'] and set(q['symbols']) == set(universe)
            for q in queries), 'Feature frozen PIT/universe mismatch')
    proof_desc = desc['input_evidence']
    require(set(proof_desc) == {'path', 'file_digest', 'input_evidence_ref'} and
            type(proof_desc['path']) is str and Path(proof_desc['path']).is_absolute() and
            file_digest(proof_desc['path']) == proof_desc['file_digest'] and
            feature['input_evidence_ref'] == proof_desc['input_evidence_ref'],
            'Feature input evidence mismatch')
    # Original saved parents can use spaced JSON. Keep the byte hash as
    # their immutable file identity and verify the logical evidence ref.
    # Canonical writer bytes avoid reserializing the largest proofs.
    try:
        byte_ref = _canonical_file_ref(proof_desc['path'])
    except ValueError:  # Valid JSON without the canonical writer's final LF.
        byte_ref = None
    proof = _read(proof_desc['path'])
    require(file_digest(proof_desc['path']) == proof_desc['file_digest'], 'Feature proof changed during read')
    require(byte_ref == feature['input_evidence_ref'] or
            digest(proof) == feature['input_evidence_ref'], 'Feature input evidence mismatch')
    by_date = {p['session']: p for p in proof}
    require(len(by_date) == len(proof) and set(by_date) == set(dates), 'Feature proof date coverage mismatch')
    indexed = grid(feature['rows'], dates, universe, 'session')
    for day in dates:
        require(day in calendar, 'outside Feature parent date')
        plan_ref = digest(by_date[day]['core_plan'])
        for security in universe:
            row = indexed[security, day]
            require(row['source_refs'] == [by_date[day]['core_frame_ref'], plan_ref], 'Feature Core source mismatch')
            require(type(row['member']) is bool and all(type(v) is bool for v in row['validity']) and
                    len(row['values']) == len(row['validity']) == len(row['availability']) == len(columns),
                    'Feature row schema mismatch')
            require(_instant(row['knowledge_cutoff']) == _instant(day+'T20:30:00+08:00') and
                    all(a is None or _instant(a) <= _instant(row['knowledge_cutoff']) for a in row['availability']),
                    'original Feature clock conflict')
    return feature, indexed, by_date

def load_saved_feature_inputs(manifest):
    """Validate the complete common parents once and release each full proof."""
    calendar, universe, columns = _validate_saved_manifest(manifest)
    identity = common_source_identity(manifest)
    scope = read_parent(manifest['scope'], 'scope_bundle_ref')
    for key, name in (('request_ref', 'request'), ('result_ref', 'result'), ('source_proof_ref', 'source_proof')):
        require(scope[key] == digest(scope[name]), 'scope source linkage mismatch')
    result = scope['result']
    require(result['read_sessions'] == calendar and result['read_symbols'] == universe and
            result['snapshot_id'] == manifest['snapshot'] and result['pit_policy'] == manifest['pit_policy'],
            'frozen scope projection mismatch')
    del result, scope
    rows, parents, builds = {}, {}, {}
    require(bool(manifest['feature_parents']), 'saved Feature parents required')
    for desc in manifest['feature_parents']:
        require(set(desc) == {'features', 'input_evidence', 'sessions'}, 'Feature parent descriptor required')
        feature, indexed, by_date = validate_feature_parent(desc, manifest)
        parent_ref = feature['feature_ref']
        for day in desc['sessions']:
            require(day not in parents, 'overlapping Feature parent dates')
            for security in universe:
                rows[security, day] = indexed[security, day]
            parents[day] = {'feature_ref': parent_ref, 'qlib_view_ref': feature['qlib_view']['view_id']}
        builds[parent_ref] = feature
        del by_date, indexed, feature
    require(common_source_identity(manifest) == identity, 'common source definition changed during validation')
    return _SavedFeatureInputs(_FEATURE_INPUT_TOKEN, identity=identity, rows=rows,
        parents=parents, builds=builds)


def project_saved_fold(manifest, spec, *, feature_inputs=None, reader=read_parent):
    """Verify fold labels/clocks, return the original six selected stage values.

    A validated common object avoids reading its scope, Features, and proofs
    again. The reader hook is for the batch's private verified label cache.
    """
    calendar, universe, columns = _validate_saved_manifest(manifest)
    training_dates, prediction_dates = validate_spec(spec, calendar)
    if feature_inputs is None:
        feature_inputs = load_saved_feature_inputs(manifest)
    data = _feature_input_data(feature_inputs)
    require(data['identity'] == common_source_identity(manifest), 'saved common source identity mismatch')
    rows, parents = data['rows'], data['parents']
    needed = set(training_dates + prediction_dates)
    require(needed <= set(parents), 'missing complete Feature date')
    compact = spec['contract_version'] == 'stock_ml_fold_spec_v2'
    feature_slice = seal({'contract_version': 'stock_feature_slice_v2' if compact else 'stock_feature_slice_v1',
        'input_manifest_ref': digest(manifest),
        'universe': universe, 'ordered_features': columns, 'catalog_ref': manifest['catalog_ref'],
        'selection': manifest['feature_selection'], 'training_sessions': training_dates,
        'prediction_sessions': prediction_dates,
        'parents_by_session': {d: deepcopy(parents[d]) for d in sorted(needed)},
        # v2 binds the full training range to immutable parents. Only OOS rows
        # are materialized here; Dataset records the actual joined training keys.
        'rows': [deepcopy(rows[s, d]) for d in (prediction_dates if compact else sorted(needed))
                 for s in universe]}, 'feature_ref')
    # A complete Raw ref binds its query and horizon. Reuse only this call's
    # admitted object/index at the exact fit information set, never selected
    # normalized rows or a caller-supplied trusted flag.
    raw_admissions = {}
    def raw_parent(descriptor):
        key = digest({'descriptor':descriptor,'snapshot':manifest['snapshot'],
            'pit_policy':manifest['pit_policy'],'calendar':manifest['calendar'],
            'universe':manifest['universe'],'fit_cutoff':_instant(spec['fit_cutoff']).isoformat()})
        def signature(value):
            return digest({'raw_ref':value['label_ref'],'query':value['source_evidence']['context'],
                           'label_spec':value['label_spec'],'calendar_ref':value['calendar_ref']})
        if key not in raw_admissions:
            before = file_fingerprint(descriptor['path'])
            raw = reader(descriptor, 'label_ref')
            require(raw['label_ref'] == descriptor['label_ref'], 'Raw parent ref mismatch')
            indexed = validate_raw(raw, manifest, spec['fit_cutoff'])
            require(file_fingerprint(descriptor['path']) == before, 'Raw parent changed during admission')
            raw_admissions[key] = raw, indexed, before, signature(raw), descriptor['path']
        raw, indexed, mark, admitted, path = raw_admissions[key]
        require(file_fingerprint(path) == mark and signature(raw) == admitted,
                'Raw parent changed during projection')
        return raw, indexed
    norm_rows, label_parents, section_refs, raw_refs = {}, [], [], []
    for desc in manifest['training_labels']:
        require(set(desc) == {'raw', 'normalized', 'sessions', 'raw_projection'}, 'training label descriptor required')
        dates = ordered(desc['sessions'], 'normalized parent dates')
        raw, raw_index = raw_parent(desc['raw'])
        norm = reader(desc['normalized'], 'label_ref'); _verify_normalized_labels(norm)
        require(type(desc['raw_projection']) is bool, 'explicit raw projection flag required')
        projected = raw
        if desc['raw_projection']:
            projected = {k: v for k, v in raw.items() if k not in ('rows', 'label_ref')}
            projected.update(rows=[r for r in raw['rows'] if r['feature_session'] in dates],
                parent_label_ref=raw['label_ref'], feature_parent_ref=norm['feature_ref'], date_projection=dates)
            projected = seal(projected, 'label_ref')
        require(norm['contract_version'] == 'stock_normalized_label_build_v1' and
                norm['raw_label_ref'] == projected['label_ref'] and norm['normalization_spec'] == NORMALIZATION_SPEC and
                _instant(norm['cutoff']) == _instant(spec['fit_cutoff']), 'normalized parent input/cutoff mismatch')
        indexed = grid(norm['rows'], dates, universe, 'feature_session')
        require([s['feature_session'] for s in norm['sections']] == dates, 'normalized section coverage mismatch')
        for section, plan, facts, context, frame in zip(norm['sections'], norm['core_plan'],
                norm['core_facts'], norm['core_context'], norm['core_frames']):
            day = section['feature_session']
            require(day in parents and norm['feature_ref'] == parents[day]['feature_ref'], 'normalized Feature parent mismatch')
            parts = normalization_section_inputs(projected, feature_ref=norm['feature_ref'], feature_rows=rows,
                session=day, securities=universe, width=len(columns), cutoff=norm['cutoff'], raw_index=raw_index)
            require(plan == parts['plan'] and facts == parts['facts'] and context == parts['context'],
                    'normalized Core/raw input linkage mismatch')
            require(frame['schema'] == [plan['outputs'][0]['column']] and
                    frame['recipe_ref'] == plan['recipe_ref'] and frame['calendar_ref'] == raw['calendar_ref'] and
                    frame['source_bindings'] == plan['sources'], 'normalized frame metadata mismatch')
            computed = grid(frame['rows'], [day], universe, 'session')
            eligible = []
            for security in universe:
                key = security, day; row = indexed[key]; raw_row = raw_index[key]
                if day not in needed:
                    continue
                require(key not in norm_rows, 'overlapping normalized dates')
                original_reason = parts['reasons'][key]
                if original_reason is None:
                    eligible.append([security, day])
                wire = computed[key]
                valid = original_reason is None and wire['valid'][0] and _finite(wire['values'][0])
                require(row['valid'] is valid and row['raw_return'] == raw_row['return'] and
                        row['invalid_reason'] == (None if valid else original_reason or 'NORMALIZATION_UNDEFINED') and
                        row['label_available_at'] == raw_row['label_available_at'] and
                        row['end_session'] == raw_row['end_session'] and row['start_session'] == raw_row['start_session'] and
                        row['normalized_target'] == (wire['values'][0] if valid else None) and
                        row['normalized_available_at'] == (wire['availability'][0] if valid else None),
                        'normalized saved row/Core/raw mismatch')
                require(_instant(wire['cutoff']) == _instant(context['cutoffs'][day]) and
                        row['source_refs'] == sorted(set(raw_row['source_refs'] + [projected['label_ref'],
                            norm['feature_ref'], section['section_ref'], section['frame_ref']])),
                        'normalized output provenance/clock mismatch')
                if valid:
                    require(_instant(row['normalized_available_at']) <= _instant(spec['fit_cutoff']), 'normalized clock exceeds fit')
                norm_rows[key] = {**row, 'raw_parent_ref': raw['label_ref'], 'normalized_parent_ref': norm['label_ref']}
            if day in needed:
                require(section['eligible_keys'] == eligible, 'normalized eligibility linkage mismatch')
                section_refs.append(section['section_ref'])
        label_parents.append({'raw_label_ref': raw['label_ref'], 'projected_raw_label_ref': projected['label_ref'],
            'label_ref': norm['label_ref'], 'feature_ref': norm['feature_ref'], 'cutoff': norm['cutoff'], 'sessions': dates})
        raw_refs.append(raw['label_ref'])
        del raw, raw_index, norm, indexed, computed, projected
    training, excluded, label_rows = [], {}, []
    for day in training_dates:
        i = calendar.index(day)
        immature = i+5 >= len(calendar) or calendar[i+5] > _instant(spec['fit_cutoff']).date().isoformat()
        for security in universe:
            key = security, day; feature = rows[key]; label = norm_rows.get(key)
            require(immature or label is not None, 'missing mature normalized label date')
            reason = 'LABEL_NOT_MATURE' if immature else (label['invalid_reason'] or 'NORMALIZATION_UNDEFINED') if not label['valid'] else None
            if reason is None:
                require(label['valid'] is True and _finite(label['normalized_target']), 'training target must be valid and finite')
                require(feature_available(feature) is not None and
                        _instant(feature['knowledge_cutoff']) <= _instant(spec['fit_cutoff']) and
                        _instant(feature_available(feature)) <= _instant(spec['fit_cutoff']), 'training Feature clock exceeds fit')
                training.append({**deepcopy(feature), 'label': label['normalized_target'], 'raw_return': label['raw_return'],
                    'label_available_at': label['label_available_at'], 'normalized_available_at': label['normalized_available_at']})
            else:
                excluded[reason] = excluded.get(reason, 0)+1
            label_rows.append({'security_id': security, 'feature_session': day,
                'normalized_parent_ref': label['normalized_parent_ref'] if label else None,
                'valid': reason is None, 'invalid_reason': reason})
    label_selection = ({'training_sessions': training_dates, 'selection_ref': digest(label_rows),
                        'selection_count': len(label_rows), 'excluded': excluded} if compact else {'rows': label_rows})
    label_slice = seal({'contract_version': 'stock_fold_label_slice_v2' if compact else 'stock_fold_label_slice_v1',
        'parents': label_parents, 'cutoff': spec['fit_cutoff'], **label_selection,
        'section_refs': section_refs}, 'label_ref')
    evaluation = reader(manifest['evaluation_labels'], 'label_ref')
    evaluation_index = validate_raw(evaluation, manifest, spec['evaluation_cutoff'])
    require(all((s, d) in evaluation_index for d in prediction_dates for s in universe), 'missing OOS raw label grid')
    require(all(file_fingerprint(path) == mark and
                digest({'raw_ref':raw['label_ref'],'query':raw['source_evidence']['context'],
                        'label_spec':raw['label_spec'],'calendar_ref':raw['calendar_ref']}) == admitted
                for raw, indexed, mark, admitted, path in raw_admissions.values()),
            'Raw parent changed before projection completed')
    return feature_slice, label_slice, training, excluded, sorted(set(raw_refs)), evaluation

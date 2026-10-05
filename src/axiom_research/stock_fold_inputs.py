"""Validate explicit saved parents and select a fold; no numerical executor.

Parents are read and released one at a time. This is bounded saved projection,
not a streaming proof loader or a general storage/path migration service.
"""
from copy import deepcopy
from pathlib import Path

from .stock_artifacts import (digest, file_digest, _read, _verify_ref,
                              _canonical_file_ref, _verify_normalized_labels)
from .stock_label_contracts import (_instant, _session, _finite,
                                    NORMALIZATION_SPEC, normalization_section_inputs)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seal(value, key):
    return {**value, key: digest(value)}


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


def validate_spec(spec, calendar):
    require(set(spec) == {'contract_version', 'training_window', 'fit_session', 'fit_cutoff',
        'simulated_model_available_at', 'oos_trade_sessions', 'inference_cutoff_by_session',
        'evaluation_cutoff'} and spec['contract_version'] == 'stock_ml_fold_spec_v1', 'unsupported fold spec')
    window = spec['training_window']
    require(window == {'unit': 'feature_sessions', 'length': 65, 'end': 'previous_fit_session'} and
            type(window['length']) is int, 'this bounded profile requires 65 actual feature sessions')
    fit = spec['fit_session']; require(fit in calendar, 'fit session outside frozen calendar')
    i = calendar.index(fit); require(i >= 65, 'insufficient frozen training calendar')
    training = calendar[i-65:i]
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


def project_saved_fold(manifest, spec):
    """Verify full immutable parents, return only selected rows and stage slices."""
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
    scope = read_parent(manifest['scope'], 'scope_bundle_ref')
    for key, name in (('request_ref', 'request'), ('result_ref', 'result'), ('source_proof_ref', 'source_proof')):
        require(scope[key] == digest(scope[name]), 'scope source linkage mismatch')
    result = scope['result']
    require(result['read_sessions'] == calendar and result['read_symbols'] == universe and
            result['snapshot_id'] == manifest['snapshot'] and result['pit_policy'] == manifest['pit_policy'],
            'frozen scope projection mismatch')
    del scope
    training_dates, prediction_dates = validate_spec(spec, calendar)
    needed = set(training_dates + prediction_dates)
    needed_for_validation = needed | {d for desc in manifest['training_labels'] for d in desc['sessions']}
    rows, parents = {}, {}
    require(bool(manifest['feature_parents']), 'saved Feature parents required')
    for desc in manifest['feature_parents']:
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
            require(day in calendar and day not in parents, 'overlapping/outside Feature parent dates')
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
                if day in needed_for_validation:
                    rows[security, day] = row
            parents[day] = {'feature_ref': parent_ref, 'qlib_view_ref': feature['qlib_view']['view_id']}
        del proof, by_date, indexed, feature
    require(needed <= set(parents), 'missing complete Feature date')
    feature_slice = seal({'contract_version': 'stock_feature_slice_v1', 'input_manifest_ref': digest(manifest),
        'universe': universe, 'ordered_features': columns, 'catalog_ref': manifest['catalog_ref'],
        'selection': manifest['feature_selection'], 'training_sessions': training_dates,
        'prediction_sessions': prediction_dates, 'parents_by_session': {d: parents[d] for d in sorted(needed)},
        'rows': [rows[s, d] for d in sorted(needed) for s in universe]}, 'feature_ref')
    norm_rows, label_parents, section_refs, raw_refs = {}, [], [], []
    for desc in manifest['training_labels']:
        require(set(desc) == {'raw', 'normalized', 'sessions', 'raw_projection'}, 'training label descriptor required')
        dates = ordered(desc['sessions'], 'normalized parent dates')
        raw = read_parent(desc['raw'], 'label_ref'); raw_index = validate_raw(raw, manifest, spec['fit_cutoff'])
        norm = read_parent(desc['normalized'], 'label_ref'); _verify_normalized_labels(norm)
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
                training.append({**feature, 'label': label['normalized_target'], 'raw_return': label['raw_return'],
                    'label_available_at': label['label_available_at'], 'normalized_available_at': label['normalized_available_at']})
            else:
                excluded[reason] = excluded.get(reason, 0)+1
            label_rows.append({'security_id': security, 'feature_session': day,
                'normalized_parent_ref': label['normalized_parent_ref'] if label else None,
                'valid': reason is None, 'invalid_reason': reason})
    label_slice = seal({'contract_version': 'stock_fold_label_slice_v1', 'parents': label_parents,
        'cutoff': spec['fit_cutoff'], 'rows': label_rows, 'section_refs': section_refs}, 'label_ref')
    evaluation = read_parent(manifest['evaluation_labels'], 'label_ref')
    evaluation_index = validate_raw(evaluation, manifest, spec['evaluation_cutoff'])
    require(all((s, d) in evaluation_index for d in prediction_dates for s in universe), 'missing OOS raw label grid')
    return feature_slice, label_slice, training, excluded, sorted(set(raw_refs)), evaluation

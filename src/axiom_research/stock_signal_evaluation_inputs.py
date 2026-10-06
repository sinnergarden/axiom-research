"""Stdlib source admission and exact key selection for saved Signal evaluation."""
from copy import deepcopy
from pathlib import Path

from .stock_artifacts import digest, file_digest, _read, _verify_ref
from .stock_label_contracts import _instant, _session, _finite
from .stock_fold_inputs import file_fingerprint


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _ordered(values, name):
    _require(type(values) is list and bool(values) and
        all(type(v) is str and v for v in values) and values == sorted(set(values)),
        'ordered unique ' + name + ' required')
    return values


def _scope(value):
    _require(type(value) is dict and set(value) ==
        {'sessions', 'universe', 'evaluation_cutoff', 'calendar'}, 'explicit evaluation scope required')
    out = deepcopy(value)
    for name in ('sessions', 'calendar'):
        for day in _ordered(out[name], name):
            _session(day)
    _ordered(out['universe'], 'universe')
    _require(set(out['sessions']) <= set(out['calendar']), 'evaluation sessions outside calendar')
    out['evaluation_cutoff'] = _instant(out['evaluation_cutoff']).isoformat().replace('+00:00', 'Z')
    return out


def _descriptor(value, ref_key):
    _require(type(value) is dict and set(value) == {'path', 'file_digest', ref_key},
        'explicit saved input descriptor required')
    _require(type(value['path']) is str and Path(value['path']).is_absolute(),
        'fixed absolute saved input path required')
    _require(file_digest(value['path']) == value['file_digest'], 'saved input file digest mismatch')
    wire = _read(value['path'])
    # Original experiments save their two Raw Label builds inside one bundle.
    if ref_key == 'label_ref' and wire.get('contract_version') == 'stock_label_bundle_v1':
        _verify_ref(wire, 'label_ref')
        candidates = [wire[name] for name in ('training', 'evaluation')
                      if wire[name].get('label_ref') == value['label_ref']]
        _require(len(candidates) == 1, 'ambiguous/missing saved Raw Label build')
        wire = candidates[0]
    _verify_ref(wire, ref_key)
    _require(wire[ref_key] == value[ref_key], 'saved input content ref mismatch')
    _require(file_digest(value['path']) == value['file_digest'], 'saved input changed during read')
    return wire


def _source_records(root):
    """Freeze the existing record files, including original fold parent paths."""
    manifest = _read(root/'manifest.json')
    files = {root/'manifest.json', *(root/name for name in manifest['files'])}
    expected = {}
    # A fold's input manifest already holds every external parent descriptor.
    if (root/'fold.json').is_file():
        definition = _read(root/'fold.json')['definition']['input_manifest']
        def visit(item):
            if type(item) is dict:
                if {'path', 'file_digest'} <= set(item):
                    path = Path(item['path'])
                    _require(path not in expected or expected[path] == item['file_digest'],
                             'conflicting saved source parent descriptor')
                    expected[path] = item['file_digest']
                    files.add(path)
                for child in item.values():
                    visit(child)
            elif type(item) is list:
                for child in item:
                    visit(child)
        visit(definition)
    # A Raw descriptor repeats in every normalized shard. Hash each unique
    # path once in this call, reject conflicting refs, and check the entire
    # source set before/after hashing. No admission survives this call.
    marks = {path: file_fingerprint(path) for path in files}
    records = [{'path': str(path), 'file_digest': file_digest(path)}
               for path in sorted(files, key=str)]
    observed = {Path(row['path']): row['file_digest'] for row in records}
    _require(all(observed[path] == ref for path, ref in expected.items()), 'saved source parent mismatch')
    _require(all(file_fingerprint(path) == mark for path, mark in marks.items()),
             'saved source changed during hashing')
    return records


def _grid(rows, universe, dates, date_key, label):
    expected = {(security, day) for day in dates for security in universe}
    indexed = {}
    _require(type(rows) is list, label + ' rows required')
    for row in rows:
        key = row.get('security_id'), row.get(date_key)
        _require(key in expected and key not in indexed, 'duplicate/unexpected ' + label + ' key')
        indexed[key] = row
    _require(set(indexed) == expected, 'complete ' + label + ' grid required')
    return indexed


def _query(query, *, snapshot, pit, universe, cutoff, purpose=None):
    _require(type(query) is dict and query.get('pit_policy') == pit and
        set(query.get('symbols', ())) == set(universe), 'saved query PIT/universe mismatch')
    if purpose is not None:
        _require(query.get('purpose') == purpose, 'saved query purpose mismatch')
    days = _ordered(query.get('sessions'), 'query sessions')
    clocks = query.get('cutoff_by_session')
    _require(type(clocks) is dict and set(clocks) == set(days) and
        all(_instant(clock) <= cutoff for clock in clocks.values()), 'saved source query exceeds evaluation cutoff')


def _feature_proof(evidence, snapshot, pit, universe, cutoff):
    sources = evidence.get('source_evidence')
    _require(type(sources) is dict and bool(sources), 'Feature saved source evidence required')
    members = []
    for source in sources.values():
        context = source['query_context']
        _require(context.get('snapshot_id') == snapshot, 'Feature proof Snapshot mismatch')
        _query(context['query'], snapshot=snapshot, pit=pit, universe=universe,
            cutoff=cutoff, purpose='decision_facts')
        _require(context['query']['sessions'] == evidence['sessions'] and
            set(context['query']['cutoff_by_session']) == set(evidence['cutoffs']) and
            all(_instant(context['query']['cutoff_by_session'][day]) == _instant(clock)
                for day, clock in evidence['cutoffs'].items()), 'Feature proof query scope/clock mismatch')
        if source['field'] == 'is_member':
            _require(context.get('domain') == 'universe_membership' and
                context['query']['fields'] == ['is_member'] and source.get('batch_ref') == evidence['membership_ref'],
                'Feature original membership source binding mismatch')
            members.append(source)
        elif source.get('batch_ref') is not None:
            _require(source['batch_ref'] == evidence['adjusted_input_ref'], 'Feature original adjusted source binding mismatch')
    _require(bool(members), 'Feature historical membership source required')


def _saved_signal(descriptor, scope, *, batch=None):
    signal = _descriptor(descriptor, 'signal_run_ref')
    root = Path(descriptor['path']).parent
    _require(Path(descriptor['path']).name == 'predictions.json', 'existing saved predictions.json required')
    version = signal.get('contract_version')
    if version == 'stock_prediction_run_v1':
        _require(batch is None, 'saved batch requires fold Signals')
        from .stock_artifacts import load_stock_ml_experiment
        load_stock_ml_experiment(root)
        record = _read(root/'experiment.json')
        config = record['definition']['config']
        feature = _read(root/'features.json')
        model = _read(root/'model.json')
        proof = _read(root/'feature-inputs.json')
        by_day = {p['session']: p for p in proof}
        _require(len(by_day) == len(proof), 'duplicate Feature proof day')
        calendar, snapshot, pit = config['calendar'], config['snapshot'], config['pit_policy']
        dates = config['prediction_sessions']
    elif version == 'stock_prediction_run_v2':
        from .stock_fold_artifacts import load_stock_ml_fold
        load_stock_ml_fold(root, batch=batch)
        record = _read(root/'fold.json')
        config = record['definition']['input_manifest']
        feature = _read(root/'feature-slice.json')
        model = _read(root/'model.json')
        calendar, snapshot, pit = config['calendar'], config['snapshot'], config['pit_policy']
        dates = feature['prediction_sessions']
    else:
        raise ValueError('unsupported saved stock Signal contract')
    _require(calendar == scope['calendar'], 'saved Signal frozen calendar mismatch')
    _require(type(snapshot) is str and snapshot not in ('', 'latest', 'current') and
        type(pit) is str and bool(pit), 'fixed saved Snapshot/PIT required')
    _require(set(scope['universe']) <= set(signal['universe']), 'evaluation universe outside saved Signal')
    _require(signal.get('score_unit') == 'dimensionless' and
        signal.get('signal_stage') == 'prediction_raw' and
        signal.get('score_semantics') == model.get('target_semantics') and
        signal.get('feature_ref') == feature['feature_ref'] and
        signal.get('model_ref') == model['model_ref'], 'saved Signal stage/unit linkage mismatch')
    universe = signal['universe']; _ordered(universe, 'saved Signal universe')
    feature_dates = sorted({r['session'] for r in feature['rows']})
    _require(set(feature_dates) <= set(calendar) and set(dates) <= set(feature_dates),
        'saved Feature/Signal date scope mismatch')
    feature_index = _grid(feature['rows'], universe, feature_dates, 'session', 'Feature')
    indexed = _grid(signal['rows'], universe, dates, 'session', 'Signal')
    cutoff = _instant(scope['evaluation_cutoff'])
    _require(_instant(model['fit_cutoff']) <= cutoff, 'saved model fit exceeds evaluation cutoff')
    if version == 'stock_prediction_run_v1':
        view = feature['qlib_view']
        _require(view.get('snapshot_id') == snapshot, 'saved Feature Snapshot mismatch')
        for query in view['queries'] + [view['universe_query']]:
            _query(query, snapshot=snapshot, pit=pit, universe=universe, cutoff=cutoff)
        _require(set(by_day) == set(feature_dates), 'Feature proof scope mismatch')
        for day, evidence in by_day.items():
            _require(evidence['sessions'] == sorted(evidence['cutoffs']) and
                all(_instant(c) <= cutoff for c in evidence['cutoffs'].values()), 'Feature proof query cutoff mismatch')
            _require(evidence.get('membership_ref') and evidence.get('adjusted_input_ref'), 'Feature membership/source proof required')
            _feature_proof(evidence, snapshot, pit, universe, cutoff)
    else:
        by_day = {}
        wanted = set(feature_dates)
        for parent in config['feature_parents']:
            if not wanted.intersection(parent['sessions']):
                continue  # Complete parent dates/proofs were admitted by the fold loader.
            for evidence in _read(parent['input_evidence']['path']):
                if evidence['session'] not in wanted:
                    continue
                _feature_proof(evidence, snapshot, pit, universe, cutoff)
                _require(evidence['session'] not in by_day, 'duplicate Feature source proof day')
                by_day[evidence['session']] = evidence
    for key, row in feature_index.items():
        _require(type(row.get('member')) is bool, 'saved historical membership required')
        knowledge = _instant(row['knowledge_cutoff'])
        _require(knowledge.date().isoformat() >= key[1] and knowledge <= cutoff and
            all(a is None or _instant(a) <= knowledge for a in row['availability']), 'Feature source clock conflict')
        proof_row = by_day[key[1]]
        _require(all(_instant(clock) <= knowledge for clock in proof_row['cutoffs'].values()),
            'Feature original source query exceeds knowledge cutoff')
        members = proof_row['core_plan'].get('reference_members')
        _require(type(members) is dict and key[1] in members and
            row['member'] is (key[0] in members[key[1]]), 'Feature original historical membership mask mismatch')
        if version == 'stock_prediction_run_v1':
            _require(knowledge == _instant(config['cutoff_by_session'][key[1]]),
                'Feature original config clock mismatch')
            _require(row['source_refs'] == [proof_row['core_frame_ref'], digest(proof_row['core_plan'])],
                'Feature original source refs mismatch')
    for key, row in indexed.items():
        original = feature_index[key]
        knowledge, available = _instant(row['knowledge_cutoff']), _instant(row['available_at'])
        _require(knowledge.date().isoformat() >= key[1] and available <= knowledge <= cutoff,
            'Signal source clock conflict')
        _require(row.get('member') is original['member'] and type(row.get('valid')) is bool,
            'Signal original membership mismatch')
        if row['valid']:
            _require(row['member'] and _finite(row['score']) and row.get('invalid_reason') is None and
                all(original['validity']) and all(_finite(v) for v in original['values']),
                'Signal valid value/Feature mismatch')
        else:
            _require(row['score'] is None and bool(row.get('invalid_reason')), 'Signal invalid rows must preserve null/reason')
        if version == 'stock_prediction_run_v1':
            _require(row['source_refs'] == [feature['feature_ref'], model['model_ref'],
                feature['catalog_ref'], feature['qlib_view']['view_id']], 'Signal original source refs mismatch')
            feature_clocks = [_instant(v) for v in original['availability'] if v is not None]
            _require(knowledge == _instant(original['knowledge_cutoff']) and
                available == max(feature_clocks + [_instant(model['fit_cutoff'])]), 'Signal original clock mismatch')
    clocks = [model['fit_cutoff'], *(r['knowledge_cutoff'] for r in feature_index.values()),
              *(r['knowledge_cutoff'] for r in indexed.values())]
    if version == 'stock_prediction_run_v1':
        clocks.extend(c for q in view['queries'] + [view['universe_query']]
                      for c in q['cutoff_by_session'].values())
    return {'signal': signal, 'rows': indexed, 'members': feature_index,
        'snapshot': snapshot, 'pit': pit, 'source_records': _source_records(root),
        'metadata': {'signal_contract_version': version, 'signal_run_ref': signal['signal_run_ref'],
            'prediction_sessions': sorted(set(dates)),
            'signal_stage': signal['signal_stage'], 'score_unit': signal['score_unit'],
            'score_semantics': signal['score_semantics'], 'model': model,
            'feature_contract_version': feature.get('contract_version'), 'feature_ref': feature['feature_ref'],
            'snapshot': snapshot, 'pit_policy': pit,
            'evaluation_clock_floor': max(map(_instant, clocks)).isoformat().replace('+00:00', 'Z')}}


def _raw_labels(descriptor, scope, snapshot, pit):
    raw = _descriptor(descriptor, 'label_ref')
    _require(raw.get('contract_version') == 'stock_label_build_v1', 'evaluation requires saved Raw Labels')
    calendar = scope['calendar']; cutoff = _instant(scope['evaluation_cutoff'])
    _require(raw.get('calendar_ref') == digest({'contract_version': 'stock_label_calendar_v1', 'sessions': calendar}),
        'Raw Label frozen calendar mismatch')
    spec = raw['label_spec']
    _require(spec.get('normalization') == 'none' and type(spec.get('horizon_sessions')) is int and
        spec['horizon_sessions'] > 0 and type(spec.get('start_session_offset')) is int and
        type(spec.get('end_session_offset')) is int and spec.get('start_session_offset') == 1 and
        spec.get('end_session_offset') == spec['horizon_sessions'] and
        spec.get('label_id') == f"forward_{spec['horizon_sessions']}_session_open_close_v1" and
        spec.get('start_price') == 'open' and spec.get('end_price') == 'close',
        'Raw Label endpoint semantics required')
    ctx = raw['source_evidence']['context']
    _require(ctx.get('snapshot_id') == snapshot, 'Raw Label Snapshot mismatch')
    query = ctx['query']; universe = query['symbols']
    _require(set(scope['universe']) <= set(universe), 'evaluation universe outside Raw Labels')
    _query(query, snapshot=snapshot, pit=pit, universe=universe, cutoff=cutoff, purpose='label_outcomes')
    _require(set(query['sessions']) <= set(calendar), 'Raw Label query outside calendar')
    _require(ctx.get('domain') == 'market_daily' and set(query.get('fields', ())) == {'open', 'close'} and
        query.get('price_basis') == spec.get('price_basis') == 'common_anchor_adjusted_v1',
        'Raw Label price basis/domain/fields mismatch')
    clocks = {_instant(c) for c in query['cutoff_by_session'].values()}
    _require(len(clocks) == 1, 'Raw Label requires one outcome query cutoff')
    source_cutoff = next(iter(clocks))
    derivation = ctx.get('derivation')
    _require(type(derivation) is dict and _instant(derivation['decision_cutoff']) == source_cutoff and
        derivation.get('recipe_version') == 'common_anchor_price_v1' and
        derivation.get('formula') == 'price_t * factor_t / factor_anchor' and
        derivation.get('decision_session') == query['sessions'][-1] and
        derivation.get('anchor_session') == spec.get('adjustment_anchor') == query.get('adjustment_anchor') and
        derivation['anchor_session'] in calendar and derivation['anchor_session'] <= query['sessions'][-1],
        'Raw Label source derivation/anchor mismatch')
    for name in ('price_query', 'factor_query'):
        _query(derivation[name], snapshot=snapshot, pit=pit, universe=universe,
            cutoff=cutoff, purpose='label_outcomes')
        native = derivation[name]
        expected_sessions = set(query['sessions']) | ({derivation['anchor_session']} if name == 'factor_query' else set())
        fields_ok = ({'open', 'close'} <= set(native.get('fields', ())) if name == 'price_query'
            else derivation.get('factor_domain') == 'adjustment_factors' and
                 derivation.get('factor_field') in native.get('fields', ()))
        _require(set(native['sessions']) == expected_sessions and native.get('price_basis') == 'unadjusted' and
            native.get('adjustment_anchor') is None and
            all(_instant(c) == source_cutoff for c in native['cutoff_by_session'].values()) and
            fields_ok, 'Raw Label native source query mismatch')
        _require(all((native.get('policy_by_session') or {}).get(day) ==
            (query.get('policy_by_session') or {}).get(day) for day in query['sessions']),
            'Raw Label native per-session PIT policy mismatch')
    dates = sorted({r['feature_session'] for r in raw['rows']})
    _require(set(dates) <= set(calendar) and set(scope['sessions']) <= set(dates), 'Raw Label feature scope mismatch')
    indexed = _grid(raw['rows'], universe, dates, 'feature_session', 'Raw Label')
    positions = {day: i for i, day in enumerate(calendar)}
    query_sessions = set(query['sessions'])
    for key, row in indexed.items():
        i = positions[key[1]]
        for field, offset in (('start_session', 1), ('end_session', spec['horizon_sessions'])):
            endpoint = calendar[i+offset] if i+offset < len(calendar) else None
            _require(row.get(field) == endpoint, 'Raw Label calendar endpoint conflict')
        _require(type(row.get('valid')) is bool and row.get('source_refs') == [raw['source_ref']],
            'Raw Label original source binding mismatch')
        if row['valid']:
            _require(row.get('start_session') is not None and row.get('end_session') is not None and
                row['start_session'] in query_sessions and row['end_session'] in query_sessions and
                _finite(row.get('return')) and row.get('invalid_reason') is None, 'Raw Label valid value mismatch')
            _require(_instant(row['label_available_at']) <= source_cutoff,
                'Raw Label availability exceeds saved source query')
        else:
            _require(row.get('return') is None and bool(row.get('invalid_reason')), 'Raw Label invalid null/reason required')
    return raw, indexed


def _admit_inputs(signal_inputs, raw_label_input, scope, *, batch=None):
    _require(isinstance(signal_inputs, dict) and bool(signal_inputs) and
        all(type(key) is str and key for key in signal_inputs), 'ordered Signal mapping required')
    projected, closures, refs, metadata = {}, {}, {}, {}
    snapshot = pit = None
    for name, descriptors in signal_inputs.items():
        metadata[name] = []
        descriptors = descriptors if type(descriptors) is list else [descriptors]
        _require(bool(descriptors), 'nonempty weekly Signal list required')
        rows, members, prediction_features, sources, signal_refs, previous = {}, {}, {}, [], [], None
        for descriptor in descriptors:
            item = _saved_signal(descriptor, scope, batch=batch)
            metadata[name].append(item['metadata'])
            if snapshot is None:
                snapshot, pit = item['snapshot'], item['pit']
            _require((snapshot, pit) == (item['snapshot'], item['pit']), 'comparison Snapshot/PIT mismatch')
            dates = sorted({key[1] for key in item['rows']})
            _require(previous is None or previous < dates[0], 'weekly Signals must be ordered and disjoint')
            previous = dates[-1]
            _require(not set(rows) & set(item['rows']), 'duplicate weekly Signal key')
            rows.update(item['rows'])
            # Feature dates can overlap across weekly folds and carry different
            # revisions. Keep the exact dependency of each disjoint prediction;
            # the merged membership map below only supplies the shared mask.
            prediction_features.update({key: item['members'][key] for key in item['rows']})
            for key, row in item['members'].items():
                _require(key not in members or members[key]['member'] is row['member'], 'weekly historical membership conflict')
                members[key] = row
            signal_refs.append(item['signal']['signal_run_ref'])
            sources.append({'signal_input': deepcopy(descriptor), 'model_ref': item['signal']['model_ref'],
                'feature_ref': item['signal']['feature_ref'], 'source_records': item['source_records']})
        projected[name] = {'rows': rows, 'members': members, 'prediction_features': prediction_features}
        closures[name] = sources; refs[name] = signal_refs
    raw, labels = _raw_labels(raw_label_input, scope, snapshot, pit)
    return {'projected': projected, 'closures': closures, 'refs': refs, 'metadata': metadata,
            'raw': raw, 'labels': labels, 'scope': scope}


def _select_inputs(admission, scope):
    """The shared exact sample/coverage selection for original and frozen inputs."""
    projected, closures, refs = (admission[k] for k in ('projected', 'closures', 'refs'))
    raw, labels = admission['raw'], admission['labels']
    keys = [(security, day) for day in scope['sessions'] for security in scope['universe']]
    historical, native, reasons, coverage = {}, {}, {}, {}
    evaluation_cutoff = _instant(scope['evaluation_cutoff'])
    for key in keys:
        membership = {item['members'][key]['member'] for item in projected.values() if key in item['members']}
        _require(len(membership) <= 1, 'comparison historical membership conflict')
        historical[key] = next(iter(membership)) if membership else None
    for name, item in projected.items():
        valid_keys, counts, daily = [], {}, []
        for day in scope['sessions']:
            day_counts = {'session': day, 'scope_key_count': len(scope['universe']), 'reference_key_count': 0,
                'mature_valid_label_count': 0, 'prediction_valid_count': 0, 'valid_pair_count': 0, 'excluded_counts': {}}
            for security in scope['universe']:
                key = security, day; label, row = labels[key], item['rows'].get(key)
                member = historical[key]
                label_ok = label['valid'] and label['end_session'] <= evaluation_cutoff.date().isoformat() and \
                    _instant(label['label_available_at']) <= evaluation_cutoff
                predict_ok = row is not None and row['valid']
                day_counts['reference_key_count'] += member is True
                day_counts['mature_valid_label_count'] += member is True and label_ok
                day_counts['prediction_valid_count'] += member is True and predict_ok
                reason = ('MEMBERSHIP_UNKNOWN' if member is None else 'NOT_MEMBER' if not member else
                    'RAW_LABEL_INVALID' if not label['valid'] else 'LABEL_NOT_MATURE' if not label_ok else
                    'SIGNAL_MISSING' if row is None else 'SIGNAL_INVALID' if not predict_ok else None)
                if reason is None:
                    valid_keys.append(key); day_counts['valid_pair_count'] += 1
                else:
                    day_counts['excluded_counts'][reason] = day_counts['excluded_counts'].get(reason, 0) + 1
                    counts[reason] = counts.get(reason, 0) + 1
            daily.append(day_counts)
        native[name] = valid_keys; reasons[name] = counts
        coverage[name] = {field: sum(d[field] for d in daily) for field in
            ('scope_key_count', 'reference_key_count', 'mature_valid_label_count', 'prediction_valid_count', 'valid_pair_count')}
        coverage[name].update(excluded_counts=counts, by_session=daily)
    common = set(native[next(iter(native))])
    for values in native.values():
        common.intersection_update(values)
    common = [key for key in keys if key in common]
    signal_keys = {name: digest({'input_signal_refs': refs[name]}) for name in refs}
    _require(len(set(signal_keys.values())) == len(signal_keys), 'duplicate comparison Signal identity')
    def table(mask):
        pairs = [{'signal_key': signal_keys[name], 'security_id': security, 'session': day,
            'score': projected[name]['rows'][security, day]['score'], 'outcome': labels[security, day]['return']}
            for name in projected for security, day in mask[name]]
        return {'contract_version': 'signal_statistics_input_v1', 'sessions': scope['sessions'],
            'signal_keys': list(signal_keys.values()), 'pairs': sorted(pairs,
                key=lambda p: (p['signal_key'], p['session'], p['security_id']))}
    mask = {'signals': [{'name': name, 'signal_key': signal_keys[name], 'input_signal_refs': refs[name]}
                       for name in projected], 'label_ref': raw['label_ref'], 'scope': scope,
        'common_keys': [list(key) for key in common],
        'native_keys': {name: [list(key) for key in native[name]] for name in projected}}
    return {'scope': scope, 'raw': raw, 'refs': refs, 'closures': closures, 'signal_keys': signal_keys,
        'mask': mask, 'common_input': table({name: common for name in projected}),
        'native_input': table(native), 'native_coverage': coverage, 'common_keys': common}


def _inputs(signal_inputs, raw_label_input, scope, *, batch=None):
    return _select_inputs(_admit_inputs(signal_inputs, raw_label_input, scope, batch=batch), scope)

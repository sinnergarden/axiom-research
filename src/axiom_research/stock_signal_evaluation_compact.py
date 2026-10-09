"""Compact source admission only; masks, storage and statistics are shared.

The owner hook admit_stock_signal_evaluation_fold(descriptor, *, batch) yields
documents (manifest/fold/model/feature-slice/predictions JSON), common axes,
evaluation_targets (original descriptor, original header, readonly rows),
source_records and source_fingerprints. Owner admission checks all saved stage
files, including dataset/labels/evidence/booster bytes, in that same lease.
Only the owner hook creates this payload; callers cannot supply a projection.
"""
from copy import deepcopy
from pathlib import Path

from .stock_artifacts import digest, _verify_ref
from .stock_label_contracts import _instant
from .stock_signal_evaluation_inputs import _grid, _require, _select_inputs, _label_source_context
from .stock_matrix_storage import BUFFER_FIELDS, shape_size
from .stock_compact_controls import grid_ranges
from .stock_signal_evaluation_matrix import (_sources_table, _compact_features, _ref,
    _target, _target_state, _finish_targets, _label_day_ref, _label_projection_ref)


def _spec(definition):
    columnar='label_definition_ref' in definition;h=definition['horizon_sessions'] if columnar else 5
    _require(type(h) is int and h>0 and definition['formula'] == f'close(f+{h}) / open(f+1) - 1' and
        type(definition['horizon_sessions']) is int and definition['horizon_sessions'] == h and
        type(definition['start_session_offset']) is int and definition['start_session_offset'] == 1 and
        type(definition['end_session_offset']) is int and definition['end_session_offset'] == h and
        definition['price_basis'] == 'common_anchor_adjusted_v1' and
        definition['missing_policy'] == 'invalid_null_preserve_grid', 'compact Raw semantics mismatch')
    return {'label_id': f'forward_{h}_session_open_close_v1', 'normalization': 'none',
        **({'label_definition_ref':definition['label_definition_ref'],'source_contract':'data_column_selection_v1'} if columnar else {}),
        **{k: definition[k] for k in ('formula', 'horizon_sessions', 'start_session_offset',
            'end_session_offset', 'price_basis', 'missing_policy', 'implementation_ref')},
        'start_price': 'open', 'end_price': 'close'}


def _context(source):
    view=source['header']['definition']['price_view']
    if view['contract_version']=='stock_label_column_price_view_v1':
        # An explicit projection of the new owner proof, never a DataBatch wire.
        return {'contract_version':'stock_column_label_context_v1','snapshot_id':view['snapshot_ref'],
            'domain':'market_daily','query':view['query_binding'],'derivation':{
                **view['adjustment_binding'],'price_query':view['input_queries']['price'],
                'factor_query':view['input_queries']['factor'],'factor_domain':'adjustment_factors'}}
    return view['context']


def _header(source, scope, common, records):
    _require(type(source) is dict and set(source) == {'descriptor', 'header'}, 'compact source fields mismatch')
    descriptor, header = source['descriptor'], source['header']
    _require(type(descriptor) is dict and set(descriptor) == {'path', 'file_digest', 'target_ref'} and
        type(descriptor['path']) is str and Path(descriptor['path']).is_absolute() and
        _ref(descriptor['file_digest']) and _ref(descriptor['target_ref']) and
        records.get(descriptor['path']) == descriptor['file_digest'],
        'compact original target byte pin mismatch')
    _require(type(header) is dict and set(header) == {'contract_version', 'definition', 'definition_ref',
        'row_count', 'reason_dictionary', 'source_dictionary', 'buffers', 'core_ref', 'cohort', 'target_ref'},
        'exact compact target header required')
    _verify_ref(header, 'target_ref'); definition = header['definition']
    columnar=header['contract_version']=='stock_compact_raw_v2'
    _require(header['contract_version'] in ('stock_compact_raw_v1','stock_compact_raw_v2') and
        header['target_ref'] == descriptor['target_ref'] and header['definition_ref'] == digest(definition) and
        definition['calendar'] == scope['calendar'] and definition['universe'] == common['universe'] and
        definition['snapshot'] == common['snapshot'], 'compact original target identity/axes mismatch')
    _require(type(header['row_count']) is int and type(definition['sessions']) is list and
        definition['sessions'] == sorted(set(definition['sessions'])) and bool(definition['sessions']) and
        set(definition['sessions']) <= set(scope['calendar']) and
        header['row_count'] == len(definition['sessions'])*len(definition['universe']), 'compact target coverage mismatch')
    types = {'values': 'float64_le', 'validity': 'bool_u8', 'availability': 'int64_le',
        'availability_validity': 'bool_u8', 'reason_codes': 'int32_le',
        'start_session': 'int32_le', 'end_session': 'int32_le', 'source_codes': 'int32_le'}
    _require(set(header['buffers']) == set(types) and header['core_ref'] is None and header['cohort'] is None,
        'compact Raw physical columns required')
    for name, descriptor in header['buffers'].items():
        _require(type(descriptor) is dict and set(descriptor) == BUFFER_FIELDS and
            type(descriptor['path']) is str and Path(descriptor['path']).is_absolute() and
            '..' not in Path(descriptor['path']).parts and
            Path(descriptor['path']).is_relative_to(Path(source['descriptor']['path']).parent) and
            descriptor['buffer_digest'] == descriptor['file_digest'] and _ref(descriptor['file_digest']) and
            descriptor['dtype'] == types[name] and descriptor['shape'] == [header['row_count']] and
            shape_size(descriptor['shape']) == header['row_count'],
            'compact Raw physical dtype/shape mismatch')
        _require(records.get(descriptor['path']) == descriptor['file_digest'], 'compact original buffer pin mismatch')
    source_view = definition['price_view']; _verify_ref(source_view, 'price_view_ref')
    if columnar:
        from .stock_column_inputs import validate_column_raw_binding
        validate_column_raw_binding(definition,{**common,'calendar':scope['calendar']},definition['cutoff'])
        _require(_instant(definition['cutoff'])<=_instant(scope['evaluation_cutoff']),'column outcome cutoff exceeds evaluation')
        return _spec(definition)
    _require(source_view['contract_version'] == 'stock_label_price_view_v1' and
        _ref(source_view['records_ref']) and _ref(source_view['field_meta_ref']), 'compact price-view refs required')
    context = _context(source); cutoff = _instant(scope['evaluation_cutoff'])
    _require(context['snapshot_id'] == common['snapshot'] and
        _instant(definition['cutoff']) <= cutoff, 'compact source Snapshot/visibility mismatch')
    universe, source_cutoff = _label_source_context(context, scope, common['snapshot'], common['pit_policy'],
        price_basis=definition['price_basis'], adjustment_anchor=context['query'].get('adjustment_anchor'), compact=True)
    _require(universe == definition['universe'] and source_cutoff == _instant(definition['cutoff']),
        'compact original source cutoff/calendar mismatch')
    _require(context['query']['adjustment_anchor'] == max(d for d in scope['calendar'] if d <= source_cutoff.date().isoformat()),
        'compact original adjustment anchor mismatch')
    return _spec(definition)


def _binding(spec, snapshot, label):
    # This binds an admitted compact row and its original price-view version.
    # It does not claim the legacy query-range-independent endpoint proof.
    return digest({'contract_version': 'stock_compact_evaluation_row_binding_v1',
        'label_spec': spec, 'snapshot': snapshot, 'row': label})


def _join(targets, inputs, fold_spec, common, scope, records, state, wanted):
    wanted_securities, wanted_sessions = wanted
    selector = inputs['selectors']['evaluation_labels']; _verify_ref(selector, 'selector_ref')
    _require(selector == inputs['selectors']['inference'] and
        selector['target_refs'] == [item['header']['target_ref'] for item in targets], 'compact selector/targets mismatch')
    days = []
    for item in targets:
        _require(type(item) is dict and set(item) == {'descriptor', 'header', 'rows'}, 'compact owner target fields required')
        source = deepcopy({k: item[k] for k in ('descriptor', 'header')})
        spec = _header(source, scope, common, records); header = source['header']; definition = header['definition']
        _require(state['label_spec'] is None or state['label_spec'] == spec, 'comparison evaluation Label definition/version conflict')
        state['label_spec'] = spec; ref = header['target_ref']
        _require(ref not in state['raw'] or state['raw'][ref] == source, 'conflicting compact Raw provenance')
        state['raw'][ref] = source; days.extend(definition['sessions'])
        _require(_instant(definition['cutoff']) == _instant(fold_spec['evaluation_cutoff']), 'compact evaluation vintage mismatch')
        # Only the current leased evaluation part becomes a detached list.
        indexed = _grid(list(item['rows']), common['universe'], definition['sessions'], 'feature_session', 'compact Raw')
        for key, row in indexed.items():
            _target(row, key, common['calendar'], _instant(definition['cutoff']), spec['horizon_sessions'])
            _require(row['source_refs'] == [definition['price_view']['price_view_ref']], 'compact row original source mismatch')
            if key[0] not in wanted_securities or key[1] not in wanted_sessions: continue
            binding = _binding(spec, common['snapshot'], row)
            _require(key not in state['labels'] or state['labels'][key] == row and state['leaves'][key] == binding,
                'comparison compact Label value/source/clock conflict')
            state['labels'][key] = deepcopy(row); state['leaves'][key] = binding
    _require(days == sorted(set(days)), 'compact target dates overlap or are unordered')
    lineage = {'input_ref': inputs['input_ref'], 'fold_spec_ref': digest(fold_spec),
        'fold_control': deepcopy(inputs['fold_control']), 'selector': deepcopy(selector), 'sessions': days}
    state['slices'][digest(lineage)] = lineage


def _admit_compact(signal_inputs, raw_label_input, scope, batch, *, _borrowed_lease=None, _manifest=None,
    _signals_only=False, _budget_check=None):
    from . import stock_matrix_folds as owner
    from .stock_signal_evaluation_projection import _check_marks
    hook = getattr(owner, 'admit_stock_signal_evaluation_fold', None)
    _require(callable(hook), 'compact owner admit_stock_signal_evaluation_fold interface required')
    if _borrowed_lease is not None:
        from contextlib import nullcontext
        from .stock_signal_evaluation_lease import _FoldLease
        _require(type(_borrowed_lease) is _FoldLease and _borrowed_lease._batch is batch,
                 'internal owner build lease required')
        _borrowed_lease._check_sources()
        hook = lambda descriptor, *, batch: nullcontext(_borrowed_lease)
    _require(raw_label_input is None, 'compact evaluation requires owner-admitted evaluation Raw targets')
    manifest = batch.to_dict() if _manifest is None else _manifest
    common = None; state = _target_state()
    wanted = (set(scope['universe']), set(scope['sessions']))
    folds_by_input = {}
    for item in manifest['folds']:
        ref = item['input_manifest']['input_ref']
        _require(ref not in folds_by_input, 'duplicate compact batch fold input')
        folds_by_input[ref] = item
    projected, metadata, refs, closures, records, marks = {}, {}, {}, {}, {}, {}
    for name, descriptors in signal_inputs.items():
        descriptors = descriptors if type(descriptors) is list else [descriptors]
        _require(bool(descriptors), 'nonempty ordered compact Signal list required')
        rows, members, prediction_features = {}, {}, {}; previous = None
        metadata[name], refs[name], closures[name] = [], [], []
        for descriptor in descriptors:
            with hook(descriptor, batch=batch) as lease:
                pins = _sources_table(lease.source_records)
                _require(set(lease.source_fingerprints) == set(pins), 'complete compact owner fingerprints required')
                lease_marks = {}
                for path, ref in pins.items():
                    _require(path not in records or records[path] == ref, 'conflicting compact source pin')
                    mark = tuple(lease.source_fingerprints[path]); p = Path(path)
                    _require(p not in marks or marks[p] == mark, 'compact source changed between leases')
                    records[path] = ref; marks[p] = mark; lease_marks[p] = mark
                _check_marks(lease_marks)
                documents = lease.documents; fold = documents['fold.json']; model = documents['model.json']
                features, signal = documents['feature-slice.json'], documents['predictions.json']
                _require(set(documents) == {'manifest.json', 'fold.json', 'model.json', 'feature-slice.json', 'predictions.json'},
                    'exact owner saved evaluation documents required')
                root = Path(descriptor['path']).parent; saved_manifest = documents['manifest.json']
                _require(saved_manifest['contract_version'] == 'stock_ml_fold_manifest_v2' and
                    saved_manifest['fold_ref'] == fold['fold_ref'] and set(saved_manifest['files']) == {
                        'fold.json', 'model.json', 'feature-slice.json', 'predictions.json',
                        'label-slice.json', 'dataset.json', 'signal-evidence.json', 'booster.txt'} and
                    all(pins.get(str(root/name)) == ref for name, ref in saved_manifest['files'].items()) and
                    str(root/'manifest.json') in pins, 'owner saved output pins incomplete')
                for document, key in ((fold, 'content_digest'), (model, 'model_ref'),
                        (features, 'feature_ref'), (signal, 'signal_run_ref')):
                    _verify_ref(document, key)
                inputs, spec = fold['definition']['input_manifest'], fold['definition']['fold_spec']
                _verify_ref(inputs, 'input_ref')
                original = folds_by_input.get(inputs['input_ref'])
                _require(original is not None and original['input_manifest'] == inputs and original['fold_spec'] == spec,
                    'compact Signal outside saved batch definition')
                control = inputs['fold_control']
                _require(pins.get(control['path']) == control['file_digest'], 'compact original fold control pin required')
                columnar=inputs['contract_version']=='stock_ml_saved_inputs_v5'
                _require(inputs['contract_version'] in ('stock_ml_saved_inputs_v4','stock_ml_saved_inputs_v5') and
                    fold['contract_version'] == ('stock_ml_fold_v4' if columnar else 'stock_ml_fold_v3') and
                    model['contract_version'] == ('stock_model_release_v3' if columnar else 'stock_model_release_v2') and
                    signal['contract_version'] == ('stock_prediction_run_v3' if columnar else 'stock_prediction_run_v2'), 'compact saved stage versions mismatch')
                _require(set(descriptor) == {'path', 'file_digest', 'signal_run_ref'} and
                    Path(descriptor['path']).name == 'predictions.json' and Path(descriptor['path']).is_absolute() and
                    pins.get(descriptor['path']) == descriptor['file_digest'] and
                    descriptor['signal_run_ref'] == fold['signal_run_ref'] == signal['signal_run_ref'] and
                    signal['model_ref'] == model['model_ref'] and signal['feature_ref'] == features['feature_ref'] and
                    signal['fold_spec_ref'] == digest(spec),
                    'compact saved Signal/fold/model binding mismatch')
                identity = {k: deepcopy(lease.common[k]) for k in ('snapshot', 'pit_policy', 'calendar', 'universe')}
                if columnar and not _signals_only:identity['target_spec']=deepcopy(lease.common['target_spec'])
                _require(common is None or common == identity, 'comparison compact common scope mismatch')
                common = identity
                _require(common['calendar'] == scope['calendar'] and wanted[0] <= set(common['universe']),
                    'compact frozen axes mismatch')
                days = features['prediction_sessions']; indexed = _grid(signal['rows'], common['universe'], days, 'session', 'compact Signal')
                _require(previous is None or previous < days[0], 'weekly Signals must be ordered and disjoint')
                previous = days[-1]; _require(not rows.keys() & indexed.keys(), 'duplicate weekly Signal key')
                feature_index = _grid(features['rows'], common['universe'], days, 'session', 'compact OOS Feature')
                own_members, own_features = _compact_features(feature_index)
                clocks = [model['fit_cutoff'], model['simulated_available_at']]
                for key, row in indexed.items():
                    _require(_instant(row['available_at']) <= _instant(row['knowledge_cutoff']) <= _instant(scope['evaluation_cutoff']),
                        'compact Signal source clock conflict')
                    _require(key not in members or members[key] == own_members[key], 'weekly historical membership conflict')
                    clocks.extend([row['knowledge_cutoff'], feature_index[key]['knowledge_cutoff']])
                if _budget_check is not None:
                    _budget_check([projected, metadata, refs, closures, records, marks,
                        rows, members, prediction_features], [indexed, feature_index, documents])
                rows.update(deepcopy(indexed)); members.update(own_members); prediction_features.update(own_features)
                if not _signals_only:
                    _join(lease.evaluation_targets, inputs, spec, common, scope, pins, state, wanted)
                metadata[name].append({'signal_contract_version': signal['contract_version'],
                    'signal_run_ref': signal['signal_run_ref'], 'prediction_sessions': list(days),
                    'input_ref': inputs['input_ref'], 'fold_spec_ref': digest(spec), 'fold_ref': fold['fold_ref'],
                    **{k: signal[k] for k in ('signal_stage', 'score_unit', 'score_semantics')}, 'model': deepcopy(model),
                    'feature_contract_version': features['contract_version'], 'feature_ref': features['feature_ref'],
                    'snapshot': common['snapshot'], 'pit_policy': common['pit_policy'],
                    'evaluation_clock_floor': max(map(_instant, clocks)).isoformat().replace('+00:00', 'Z')})
                refs[name].append(signal['signal_run_ref'])
                closures[name].append({'signal_input': deepcopy(descriptor), 'model_ref': model['model_ref'],
                    'feature_ref': features['feature_ref'], 'source_paths': sorted(pins)})
                _check_marks(lease_marks)
            del lease, documents, fold, model, features, signal, indexed, feature_index, own_members, own_features, row
        projected[name] = {'rows': rows, 'members': members, 'prediction_features': prediction_features}
    if _signals_only:
        batch._check_sources(); _check_marks(marks)
        return {'projected': projected, 'metadata': metadata, 'refs': refs, 'closures': closures,
            'common': common}, records, marks, manifest
    targets = _finish_targets(state, scope)
    raw = {'mode': 'compact_targets', 'label_ref': None, 'label_spec': targets['label_spec'],
        'calendar_ref': digest({'contract_version': 'stock_label_calendar_v1', 'sessions': scope['calendar']}),
        'snapshot': common['snapshot'], 'pit_policy': common['pit_policy'], 'sources': targets['raw_provenance'],
        'label_inputs': targets['label_inputs'], 'label_shard_refs': {}}
    admitted = {'projected': projected, 'metadata': metadata, 'refs': refs, 'closures': closures,
        'raw': raw, 'labels': targets['labels'], 'label_leaf_bindings': targets['label_leaf_bindings'], 'scope': scope}
    from .stock_signal_evaluation_projection import _shard
    raw['label_shard_refs'] = {day: _label_day_ref(_shard(admitted, scope, day)) for day in scope['sessions']}
    raw['label_ref'] = _label_projection_ref(raw); _select_inputs(admitted, scope)
    batch._check_sources(); _check_marks(marks)
    return admitted, records, marks, manifest


def _verify_raw(root, scope, records):
    raw = root['raw_metadata']; common = {'universe': next(iter(raw['sources'].values()))['header']['definition']['universe'],
        'snapshot': raw['snapshot'], 'pit_policy': raw['pit_policy']}
    if 'target_spec' in root['admission_receipt']['batch_manifest']['definition']:
        common['target_spec']=root['admission_receipt']['batch_manifest']['definition']['target_spec']
    _require(raw['mode'] == 'compact_targets' and root['admission_receipt']['raw_label_input'] is None and
        raw['label_ref'] == _label_projection_ref(raw), 'frozen compact Raw mode/identity mismatch')
    for ref, source in raw['sources'].items():
        _require(source['header']['target_ref'] == ref and raw['label_spec'] == _header(source, scope, common, records),
            'frozen compact Raw source/spec mismatch')
    _require(bool(raw['label_inputs']), 'complete compact target lineage required')
    batch = root['admission_receipt']['batch_manifest']
    by_input = {}
    for fold in batch['folds']:
        inputs, spec = fold['input_manifest'], fold['fold_spec']; _verify_ref(inputs, 'input_ref')
        _require(inputs['contract_version'] in ('stock_ml_saved_inputs_v4','stock_ml_saved_inputs_v5') and
            inputs['fold_spec_ref'] == digest(spec) and inputs['prepared_view'] == batch['prepared_view'] and
            inputs['input_ref'] not in by_input, 'frozen compact batch fold identity mismatch')
        by_input[inputs['input_ref']] = (inputs, spec)
    seen = set(); consumed_refs = set()
    for target in raw['label_inputs']:
        _require(set(target) == {'input_ref', 'fold_spec_ref', 'fold_control', 'selector', 'sessions'} and
            target['sessions'] == sorted(set(target['sessions'])) and
            set(target['selector']['target_refs']) <= set(raw['sources']), 'frozen compact selector lineage mismatch')
        _verify_ref(target['selector'], 'selector_ref')
        _require(target['input_ref'] in by_input and target['input_ref'] not in seen,
            'compact target lineage outside original batch fold')
        seen.add(target['input_ref']); inputs, spec = by_input[target['input_ref']]
        selector = target['selector']; days = target['sessions']; width = len(common['universe'])
        _require(target['fold_spec_ref'] == digest(spec) and target['fold_control'] == inputs['fold_control'] and
            selector == inputs['selectors']['evaluation_labels'] == inputs['selectors']['inference'] and
            selector['contract_version'] == 'stock_fold_selector_v1' and selector['kind'] == 'complete_grid' and
            set(selector) == {'contract_version', 'kind', 'row_index_ref', 'ranges', 'target_refs',
                'selected_count', 'keys_digest', 'selector_ref'} and _ref(selector['row_index_ref']) and
            type(selector['selected_count']) is int and selector['selected_count'] == len(days)*width and
            selector['keys_digest'] == digest([[s, d] for d in days for s in common['universe']]),
            'compact target lineage selector/count/keys mismatch')
        sources = [raw['sources'][ref]['header']['definition'] for ref in selector['target_refs']]
        _require(days == [day for definition in sources for day in definition['sessions']] ==
            sorted(spec['inference_cutoff_by_session']) and
            all(_instant(definition['cutoff']) == _instant(spec['evaluation_cutoff']) for definition in sources),
            'compact target lineage original date/vintage mismatch')
        # The row-index coordinate system is the prepared Feature calendar,
        # retained in the immutable batch definition rather than expanded here.
        feature_days = batch['definition']['feature_view']['spec']['feature_sessions']
        _require(selector['ranges'] == grid_ranges(days, feature_days, width), 'compact original selector ranges mismatch')
        descriptor = target['fold_control']
        _require(records.get(descriptor['path']) == descriptor['file_digest'], 'frozen original fold control pin mismatch')
        consumed_refs.update(selector['target_refs'])
    _require(set(raw['sources']) == consumed_refs, 'compact Raw sources must equal consumed evaluation targets')
    return by_input


def _label_context(root):
    sources = {day: {} for day in root['scope']['sessions']}
    for source in root['raw_metadata']['sources'].values():
        definition = source['header']['definition']; context = _context(source)
        compiled = (source, _instant(definition['cutoff']), set(context['query']['sessions']))
        for day in definition['sessions']:
            if day in sources: sources[day][definition['price_view']['price_view_ref']] = compiled
    return {'sources': sources, 'positions': {day: i for i, day in enumerate(root['scope']['calendar'])}}

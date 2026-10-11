"""Explicit original owners plus one independent, frozen outcome projection."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import os
import tempfile

from .stock_artifacts import digest, file_digest, write_json, _verify_ref
from .stock_fold_inputs import require
from .stock_label_contracts import _instant
from .stock_compact_store import OwnedStore, limits as owner_limits, _size, _view_data
from .stock_signal_evaluation_inputs import _scope
from .stock_signal_evaluation_matrix import (_sources_table, _target, _label_day_ref,
    _clock_floor_matrix, RAW_METADATA_FIELDS, _ref)

VERSION = 'stock_signal_evaluation_inputs_v6'
RECEIPT_FIELDS = {'contract_version', 'signal_inputs', 'raw_label_input', 'scope',
    'source_closure', 'source_records', 'validation_sources', 'receipt_ref',
    'owner_manifests', 'prediction_owner_refs', 'label_manifest', 'limits'}


def _owners(signals, owners, bindings, budget, retained=0, source_retained=0):
    from .stock_batch import _data
    require(type(owners) is dict and bool(owners) and type(bindings) is dict,
        'explicit owner_batches and prediction_owner_refs required')
    manifests, states = {}, {}
    for ref, batch in owners.items():
        state = _data(batch).get('matrix_state')
        require(state is not None and state.compact and not state.incomplete and state.active == 0,
            'idle COMPLETE compact owner required')
        require(state.batch['status'] == 'COMPLETE' and state._batch_ref == ref,
            'owner mapping must use the original batch_ref')
        require(id(state) not in {id(s) for s in states.values()}, 'duplicate owner instance')
        states[ref] = state
    resident, source = _live(states)
    with OwnedStore(budget, shared_bytes=resident+retained, shared_source_bytes=source+source_retained) as store:
        store.reserve(_size([s.batch for s in states.values()], maximum=store.maximum_matrix_bytes,
            retained=store.shared_bytes)*2+65536)
        require(source+source_retained <= budget['maximum_source_bytes'], 'joint source byte budget exceeded')
        manifests = {ref: batch.to_dict() for ref, batch in owners.items()}
    consumed = {}
    for name, items in signals.items():
        require(type(name) is str and bool(name), 'nonempty signal alias required')
        items = items if type(items) is list else [items]
        require(bool(items), 'nonempty ordered prediction list required')
        for item in items:
            require(type(item) is dict and set(item) == {'path', 'file_digest', 'signal_run_ref'} and
                item['signal_run_ref'] in bindings and bindings[item['signal_run_ref']] in owners,
                'missing or wrong prediction owner binding')
            ref = item['signal_run_ref']
            require(ref not in consumed or consumed[ref] == item, 'conflicting prediction reference')
            consumed[ref] = item
    require(set(bindings) == set(consumed) and set(bindings.values()) == set(owners),
        'owner mappings must exactly cover consumed predictions')
    return manifests, states


def _live(states):
    # Deduplicate real stores, never equal content digests of different copies.
    stores = {id(s.store): s.store for s in states.values()}
    for state in states.values():
        store = _view_data(state.feature, check=False)['store']; stores[id(store)] = store
    resident = sum(s.resident_bytes+s.lease_bytes for s in stores.values())
    resident += sum(s._fixed_shared_bytes for s in states.values())
    source = sum(s.metrics['source_bytes'] for s in stores.values())
    source += sum(s._fixed_shared_source_bytes for s in states.values())
    return resident, source


@contextmanager
def _joint_owner(state, states, retained, budget, source_retained=0):
    resident, source = _live(states)
    own_resident = state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes
    own_source = state.store.shared_source_bytes+state.store.metrics['source_bytes']
    require(type(retained) is int and retained >= 0, 'incremental retained byte charge required')
    extra = max(0, resident-own_resident)+retained
    source_extra = max(0, source-own_source)+source_retained
    old_limits = state.store.limits
    state.store.limits = {k: min(old_limits[k], budget[k]) for k in budget}
    state._fixed_shared_bytes += extra; state._fixed_shared_source_bytes += source_extra
    local_charge = 0
    try:
        state._sync_shared(); state.store.reserve(0)
        require(state.store.shared_source_bytes+state.store.metrics['source_bytes'] <=
            state.store.limits['maximum_source_bytes'], 'joint source byte budget exceeded')
        def check(graph, workspace):
            nonlocal local_charge
            charge = _size(graph, maximum=budget['maximum_matrix_bytes'],
                retained=state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes)
            state._fixed_shared_bytes += charge-local_charge; local_charge = charge
            state._sync_shared()
            state.store.reserve(_size(workspace, maximum=budget['maximum_matrix_bytes'],
                retained=state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes)*3+65536)
        yield check
    finally:
        state._fixed_shared_bytes -= extra+local_charge
        state._fixed_shared_source_bytes -= source_extra
        state.store.limits = old_limits; state._sync_shared()


def _merge_pins(records, marks, new_records, new_marks):
    for path, ref in new_records.items():
        require(path not in records or records[path] == ref, 'conflicting original source pin')
        if path not in records: records[path] = ref
    for path, mark in new_marks.items():
        path = Path(path)
        require(path not in marks or marks[path] == tuple(mark), 'original source changed between owners')
        if path not in marks: marks[path] = tuple(mark)


def _admit_multi(signals, raw_input, scope, owners, bindings, budget, *,
    _retained_bytes=0, _retained_source_bytes=0, _usage=None):
    from .stock_signal_evaluation_compact import _admit_compact, _binding
    from .stock_evaluation_labels import _read_manifest, _parts
    from .stock_signal_evaluation_projection import _check_marks, _shard
    manifests, states = _owners(signals, owners, bindings, budget, _retained_bytes, _retained_source_bytes)
    admitted = {'projected': {}, 'metadata': {}, 'refs': {}, 'closures': {},
        'labels': {}, 'label_leaf_bindings': {}, 'scope': scope}
    records, marks = {}, {}; common = None
    # Measure the fixed manifest/axis graph once. Fold lookup wrappers and each
    # appended piece are charged separately; never revisit the accumulated rows.
    retained = _retained_bytes+_size([admitted, records, marks, manifests],
        maximum=budget['maximum_matrix_bytes'], retained=_retained_bytes+_live(states)[0])
    indexes = {}
    with ExitStack() as operation:
        for ref, state in states.items():
            operation.enter_context(state.operation())
            with _joint_owner(state, states, retained, budget, _retained_source_bytes):
                state.store.reserve(len(manifests[ref]['folds'])*512+4096)
                index = {}
                for fold in manifests[ref]['folds']:
                    key = fold['input_manifest']['input_ref']
                    require(key not in index, 'duplicate compact batch fold input')
                    index[key] = fold
                # Values and keys are already owned by the shared manifest.
                import sys
                retained += sys.getsizeof(index)+4096
                indexes[ref] = index
        # One shared expected grid, reserved before tuple/set construction.
        with _joint_owner(state, states, retained, budget, _retained_source_bytes):
            grid_charge = len(scope['sessions'])*len(scope['universe'])*256+65536
            state.store.reserve(grid_charge)
            wanted = {(s, d) for d in scope['sessions'] for s in scope['universe']}
            retained += grid_charge
        for name, items in signals.items():
            items = items if type(items) is list else [items]
            projected = {'rows': {}, 'members': {}, 'prediction_features': {}}
            admitted['projected'][name] = projected
            for key in ('metadata', 'refs', 'closures'): admitted[key][name] = []
            previous = None
            for item in items:
                owner_ref = bindings[item['signal_run_ref']]; state = states[owner_ref]
                with _joint_owner(state, states, retained, budget, _retained_source_bytes) as check:
                    part, pins, epochs, _ = _admit_compact({name: item}, None, scope,
                        owners[owner_ref], _signals_only=True, _manifest=manifests[owner_ref], _budget_check=check,
                        _fold_index=indexes[owner_ref], _defer_source_checks=True)
                    identity = part['common']
                    require(common is None or (common['calendar'] == identity['calendar'] and
                        set(common['universe']) == set(identity['universe']) and
                        all(common[k] == identity[k] for k in ('snapshot', 'pit_policy'))),
                        'comparison original owner calendar/universe/Snapshot/PIT mismatch')
                    common = identity
                    require(common['calendar'] == scope['calendar'] and set(common['universe']) == set(scope['universe']),
                        'multi-owner requires the complete frozen universe/calendar')
                    days = part['metadata'][name][0]['prediction_sessions']
                    require(previous is None or previous < days[0], 'weekly predictions must be ordered and disjoint')
                    previous = days[-1]
                    own = part['projected'][name]
                    require(not projected['rows'].keys() & own['rows'].keys(), 'duplicate multi-owner prediction key')
                    new_records = {p:r for p,r in pins.items() if p not in records}
                    new_marks = {p:m for p,m in epochs.items() if p not in marks}
                    increment = _size([own, part['metadata'][name], part['refs'][name],
                        part['closures'][name], new_records, new_marks], maximum=budget['maximum_matrix_bytes'],
                        retained=state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes)
                    # Existing dictionaries can resize while appending. A bounded
                    # two-copy allowance protects that temporary transition.
                    state._sync_shared(); state.store.reserve(increment*2+65536)
                    retained += increment*2+65536
                    for key in projected: projected[key].update(own[key])
                    for key in ('metadata', 'refs', 'closures'): admitted[key][name].extend(part[key][name])
                    _merge_pins(records, marks, pins, epochs)
                part = own = None
            require(projected['rows'].keys() == wanted, 'each signal must cover the complete declared grid')
    resident, source_bytes = _live(states)
    with OwnedStore(budget, shared_bytes=resident+retained,
        shared_source_bytes=source_bytes+_retained_source_bytes) as store:
        store.reserve(0)
        label_manifest = _read_manifest(raw_input, store); definition = label_manifest['definition']
        require(definition['scope'] == scope and
            all(definition[k] == common[k] for k in ('snapshot', 'pit_policy')),
            'independent evaluation Label scope/Snapshot/PIT mismatch')
        raw = {'mode': 'independent_targets', 'label_ref': label_manifest['label_ref'], 'label_spec': None,
            'calendar_ref': digest({'contract_version': 'stock_label_calendar_v1', 'sessions': scope['calendar']}),
            'snapshot': common['snapshot'], 'pit_policy': common['pit_policy'], 'sources': {},
            'label_inputs': [], 'label_shard_refs': {}}
        admitted['raw'] = raw
        for descriptor, header, rows, spec in _parts(label_manifest, store):
            require(raw['label_spec'] is None or raw['label_spec'] == spec, 'independent Label definitions conflict')
            raw['label_spec'] = spec
            # Reserve detached row/source graphs before materializing them.
            charge = len(rows)*2048+_size(header, maximum=store.maximum_matrix_bytes,
                retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)*2+4096
            store.reserve(charge); store.shared_bytes += charge
            raw['sources'][header['target_ref']] = {'descriptor': deepcopy(descriptor), 'header': deepcopy(header)}
            for row in rows:
                key = row['security_id'], row['feature_session']
                _target(row, key, scope['calendar'], _instant(scope['evaluation_cutoff']), spec['horizon_sessions'])
                require(key not in admitted['labels'], 'duplicate independent evaluation Label key')
                require(row['source_refs'] == [header['definition']['price_view']['price_view_ref']],
                    'independent Label original source mismatch')
                admitted['labels'][key] = row
                admitted['label_leaf_bindings'][key] = _binding(spec, common['snapshot'], row)
        new_records = {p:r for p,r in store.hashes.items() if p not in records}
        new_marks = {Path(p):m for p,m in store.marks.items() if Path(p) not in marks}
        store.reserve(_size([new_records, new_marks], maximum=store.maximum_matrix_bytes,
            retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)*2+65536)
        _merge_pins(records, marks, new_records, new_marks)
        if _usage is not None:
            _usage['label_source_bytes'] = store.metrics['source_bytes']
        store.reserve(len(wanted)*(1024+1024*len(signals))+65536)
        raw['label_shard_refs'] = {d: _label_day_ref(_shard(admitted, scope, d)) for d in scope['sessions']}
        # Shared selector builds precisely the natural and common masks.
        from .stock_signal_evaluation_inputs import _select_inputs
        _select_inputs(admitted, scope)
        store.check()
    for batch in owners.values(): batch._check_sources()
    _check_marks(marks)
    return admitted, records, marks, manifests, label_manifest


def _root(signals, raw_input, scope, admitted, records, manifests, bindings, label_manifest, budget):
    table = [{'path': path, 'file_digest': records[path]} for path in sorted(records)]
    offsets = {r['path']: i for i, r in enumerate(table)}
    closure = {name: [{**{k: deepcopy(v) for k, v in c.items() if k != 'source_paths'},
        'source_record_indices': [offsets[p] for p in c['source_paths']]}
        for c in items] for name, items in admitted['closures'].items()}
    names = ('stock_signal_evaluation_multi.py', 'stock_evaluation_labels.py',
        'stock_signal_evaluation_compact.py', 'stock_signal_evaluation_lease.py',
        'stock_signal_evaluation_projection.py', 'stock_signal_evaluation_matrix.py',
        'stock_signal_evaluation_inputs.py', 'stock_compact_batch.py', 'stock_compact_store.py',
        'stock_matrix_folds.py', 'stock_batch.py', 'stock_artifacts.py')
    receipt = {'contract_version': 'stock_signal_evaluation_admission_v6', 'signal_inputs': deepcopy(signals),
        'raw_label_input': deepcopy(raw_input), 'scope': scope, 'source_closure': closure, 'source_records': table,
        'owner_manifests': manifests, 'prediction_owner_refs': deepcopy(bindings), 'label_manifest': label_manifest,
        'limits': budget, 'validation_sources': {n: file_digest(Path(__file__).parent/n) for n in names}}
    receipt['receipt_ref'] = digest(receipt)
    return {'contract_version': VERSION, 'scope': scope, 'signal_order': list(signals),
        'signal_refs': admitted['refs'], 'signal_metadata': admitted['metadata'], 'raw_metadata': admitted['raw'],
        'clock_floor': _clock_floor_matrix(admitted['metadata'], admitted['raw']), 'admission_receipt': receipt, 'shards': {}}


def _save_multi_inputs(signals, raw_input, scope, destination, owners, bindings, limits):
    from .contracts import ArtifactRef
    from .stock_batch import _data
    from .stock_signal_evaluation_projection import _shard, _root_id, _load_inputs, _check_marks
    budget = owner_limits(limits)
    admitted, records, marks, manifests, label_manifest = _admit_multi(signals, raw_input, scope, owners, bindings, budget)
    root = _root(signals, raw_input, scope, admitted, records, manifests, bindings, label_manifest, budget)
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.multi-owner-', dir=destination) as temporary:
        stage = Path(temporary)/'complete'; stage.mkdir()
        resident, source_bytes = _live({r: _data(b)['matrix_state']
            for r, b in owners.items()})
        with OwnedStore(budget, shared_bytes=resident+_size([admitted, root, records, marks]),
            shared_source_bytes=source_bytes) as store:
            for day in scope['sessions']:
                store.reserve(len(scope['universe'])*(4096+4096*len(signals))+65536)
                path = stage/(day+'.json'); write_json(path, _shard(admitted, scope, day))
                root['shards'][day] = {'file': path.name, 'file_digest': file_digest(path)}
            admitted = None
            root['input_id'] = _root_id(root); write_json(stage/'manifest.json', root)
            ref = ArtifactRef(artifact_type='StockSignalEvaluationInputs', artifact_id=root['input_id'],
                artifact_contract_version=VERSION, content_digest=file_digest(stage/'manifest.json'),
                uri=str(stage/'manifest.json'), metadata={'limits': budget})
            store.shared_bytes = resident+_size([root, records, marks])
            _load_inputs(ref, scope, _validate_only=True, _budget=store)
        for batch in owners.values(): batch._check_sources()
        _check_marks(marks)
        target = destination/root['input_id'][7:]
        final = ArtifactRef(artifact_type=ref.artifact_type, artifact_id=ref.artifact_id,
            artifact_contract_version=VERSION, content_digest=ref.content_digest,
            uri=str(target/'manifest.json'), metadata=ref.metadata)
        if target.exists():
            _load_inputs(final, scope, _validate_only=True)
            for batch in owners.values(): batch._check_sources()
            _check_marks(marks); return final
        try: os.rename(stage, target)
        except OSError:
            if not target.exists(): raise
            _load_inputs(final, scope, _validate_only=True)
        return final


def _verify_multi_root(root, ref, scope):
    from .stock_signal_evaluation_projection import ROOT_FIELDS, _root_id, _expand_closure
    from .stock_evaluation_labels import _identity, VERSION as LABEL_VERSION
    from .stock_signal_evaluation_compact import _header
    from .stock_target_spec import resolve_stock_label_spec
    require(type(root) is dict and set(root) == ROOT_FIELDS and root['contract_version'] == VERSION and
        root['input_id'] == ref.artifact_id == _root_id(root), 'frozen multi-owner root identity/fields mismatch')
    base = _scope(root['scope'])
    require(base == root['scope'] and scope['calendar'] == base['calendar'] and
        set(scope['sessions']) <= set(base['sessions']) and set(scope['universe']) <= set(base['universe']),
        'evaluation outside frozen multi-owner scope')
    names = root['signal_order']; receipt = root['admission_receipt']; _verify_ref(receipt, 'receipt_ref')
    require(type(names) is list and bool(names) and len(names) == len(set(names)) and
        all(type(n) is str and n for n in names) and set(names) == set(root['signal_refs']) == set(root['signal_metadata']) and
        set(receipt) == RECEIPT_FIELDS and receipt['contract_version'] == 'stock_signal_evaluation_admission_v6' and
        receipt['scope'] == base and set(receipt['signal_inputs']) == set(receipt['source_closure']) == set(names),
        'frozen multi-owner admission axes mismatch')
    require(owner_limits(receipt['limits']) == receipt['limits'] == ref.metadata.get('limits'),
        'frozen multi-owner budget binding mismatch')
    records = _sources_table(receipt['source_records']); raw = root['raw_metadata']
    label = receipt['label_manifest']; descriptor = receipt['raw_label_input']
    require(set(label) == {'contract_version', 'definition', 'raw_parts', 'label_ref'} and
        label['contract_version'] == LABEL_VERSION and label['label_ref'] == _identity(label) and
        set(descriptor) == {'path', 'file_digest', 'label_ref'} and
        descriptor['label_ref'] == label['label_ref'] and records.get(descriptor['path']) == descriptor['file_digest'],
        'frozen independent Label manifest binding mismatch')
    definition = label['definition']; target = resolve_stock_label_spec(definition['label_spec'])
    require(set(definition) == {'snapshot', 'pit_policy', 'label_spec', 'label_spec_ref', 'scope', 'implementation_ref'} and
        definition['scope'] == base and definition['label_spec_ref'] == target['label_definition_ref'] and
        _ref(definition['implementation_ref']) and set(raw) == RAW_METADATA_FIELDS and
        raw['mode'] == 'independent_targets' and raw['label_ref'] == label['label_ref'] and raw['label_inputs'] == [] and
        raw['calendar_ref'] == digest({'contract_version': 'stock_label_calendar_v1', 'sessions': base['calendar']}) and
        all(raw[k] == definition[k] for k in ('snapshot', 'pit_policy')),
        'frozen independent Label definition mismatch')
    common = {k: definition[k] for k in ('snapshot', 'pit_policy')}
    common.update(universe=base['universe'], calendar=base['calendar'], target_spec=target)
    require(type(label['raw_parts']) is list and bool(label['raw_parts']) and
        len({p['target_ref'] for p in label['raw_parts']}) == len(label['raw_parts']) == len(raw['sources']),
        'frozen independent Raw parts mismatch')
    dates = []
    for part in label['raw_parts']:
        source = raw['sources'][part['target_ref']]; header = source['header']; d = header['definition']
        require(source['descriptor'] == part and header['contract_version'] == 'stock_compact_raw_v2' and
            _header(source, base, common, records) == raw['label_spec'] and
            d['label_definition_ref'] == target['label_definition_ref'] and d['cutoff'] == base['evaluation_cutoff'],
            'frozen independent Raw source/spec mismatch')
        dates.extend(d['sessions'])
    require(dates == base['sessions'] and set(raw['label_shard_refs']) == set(base['sessions']) and
        all(_ref(r) for r in raw['label_shard_refs'].values()), 'frozen independent Label grid mismatch')
    manifests = receipt['owner_manifests']; bindings = receipt['prediction_owner_refs']; inputs = {}
    require(type(manifests) is dict and bool(manifests) and type(bindings) is dict,
        'frozen owner maps required')
    for owner_ref, manifest in manifests.items():
        _verify_ref(manifest, 'content_digest')
        require(manifest['status'] == 'COMPLETE' and manifest['contract_version'] in
            ('stock_ml_batch_inputs_v4', 'stock_ml_batch_inputs_v5') and
            manifest['batch_ref'] == owner_ref == digest({k: v for k, v in manifest.items()
                if k not in ('batch_ref', 'content_digest')}), 'frozen original owner identity mismatch')
        inputs[owner_ref] = {}
        for fold in manifest['folds']:
            inp = fold['input_manifest']; _verify_ref(inp, 'input_ref')
            require(inp['input_ref'] not in inputs[owner_ref] and inp['fold_spec_ref'] == digest(fold['fold_spec']) and
                inp['prepared_view'] == manifest['prepared_view'], 'frozen owner fold binding mismatch')
            inputs[owner_ref][inp['input_ref']] = fold
        original = manifest['definition']['feature_view']['spec']
        require(original['calendar'] == base['calendar'] and set(original['universe']) == set(base['universe']) and
            all(original[k] == raw[k] for k in ('snapshot', 'pit_policy')),
            'frozen original owner axes/Snapshot/PIT mismatch')
    expanded = _expand_closure(receipt); consumed = {}; by_day = {n: {} for n in names}
    for name in names:
        items = receipt['signal_inputs'][name]; items = items if type(items) is list else [items]
        metas = root['signal_metadata'][name]; sources = expanded[name]
        require(bool(items) and len(items) == len(metas) == len(sources) and
            [m['signal_run_ref'] for m in metas] == root['signal_refs'][name] == [d['signal_run_ref'] for d in items],
            'frozen prediction list binding mismatch')
        for item, meta, source in zip(items, metas, sources):
            signal_ref = item['signal_run_ref']
            require((signal_ref not in consumed or consumed[signal_ref] == item) and signal_ref in bindings and bindings[signal_ref] in inputs and
                source['signal_input'] == item and records.get(item['path']) == item['file_digest'],
                'frozen explicit prediction owner binding mismatch')
            consumed[signal_ref] = item; owner = inputs[bindings[signal_ref]]
            require(meta['input_ref'] in owner, 'prediction input outside original owner')
            fold = owner[meta['input_ref']]; inp, spec = fold['input_manifest'], fold['fold_spec']
            columnar = inp['contract_version'] == 'stock_ml_saved_inputs_v5'; model = meta['model']
            _verify_ref(model, 'model_ref'); pins = _sources_table(source['source_records'])
            require(inp['contract_version'] in ('stock_ml_saved_inputs_v4', 'stock_ml_saved_inputs_v5') and
                meta['fold_spec_ref'] == digest(spec) and meta['prediction_sessions'] == sorted(spec['inference_cutoff_by_session']) and
                pins.get(inp['fold_control']['path']) == inp['fold_control']['file_digest'] and _ref(meta['fold_ref']) and
                model['contract_version'] == ('stock_model_release_v3' if columnar else 'stock_model_release_v2') and
                model['clock_basis'] == 'declared_simulation' and _instant(model['fit_cutoff']) < _instant(model['simulated_available_at']) and
                meta['signal_contract_version'] == ('stock_prediction_run_v3' if columnar else 'stock_prediction_run_v2') and
                meta['feature_contract_version'] == 'stock_feature_slice_v3' and meta['feature_ref'] == source['feature_ref'] and
                model['model_ref'] == source['model_ref'] and meta['score_unit'] == 'dimensionless' and
                meta['signal_stage'] == ('raw_prediction' if columnar else 'prediction_raw') and
                meta['score_semantics'] == model['target_semantics'] and
                all(meta[k] == raw[k] for k in ('snapshot', 'pit_policy')), 'frozen original prediction stage/clock mismatch')
            for day in meta['prediction_sessions']:
                require(day in base['calendar'] and day not in by_day[name], 'frozen prediction date overlap/calendar mismatch')
                by_day[name][day] = meta
        require(set(by_day[name]) == set(base['sessions']), 'frozen prediction complete scope required')
    require(set(bindings) == set(consumed) and set(bindings.values()) == set(manifests), 'frozen owner mappings have extras')
    require(root['clock_floor'] == _clock_floor_matrix(root['signal_metadata'], raw) and
        _instant(scope['evaluation_cutoff']) >= _instant(root['clock_floor']), 'frozen evaluation source cutoff mismatch')
    require(set(root['shards']) == set(base['sessions']) and all(set(d) == {'file', 'file_digest'} and
        d['file'] == day+'.json' and _ref(d['file_digest']) for day, d in root['shards'].items()),
        'frozen multi-owner date shards mismatch')
    return by_day


def _audit_multi_input(ref):
    from .stock_batch import load_stock_ml_batch_inputs, _data
    from .stock_signal_evaluation_projection import _read_checked, _shard, _check_marks
    require(type(ref.metadata.get('limits')) is dict,
        'multi-owner frozen budget required before decoding')
    budget = owner_limits(ref.metadata['limits']); marks = {}
    with OwnedStore(budget) as ingress, ExitStack() as stack:
        # The first bytes and JSON workspace use the transport budget. Keep
        # this one root in the ordinary store cache throughout the audit.
        root = ingress.read_json({'path':str(Path(ref.uri).resolve()), 'file_digest':ref.content_digest})
        ingress.reserve(_size(root, maximum=budget['maximum_matrix_bytes'],
            retained=ingress.resident_bytes)*4+65536)
        _verify_multi_root(root, ref, root['scope'])
        receipt = root['admission_receipt']; owners, states = {}, {}
        for owner_ref, manifest in receipt['owner_manifests'].items():
            resident, source = _live(states)
            ingress.shared_bytes = resident; ingress.shared_source_bytes = source
            # The public loader copies the supplied control graph before it
            # opens its store; reserve that copy before calling it.
            ingress.reserve(_size(manifest, maximum=budget['maximum_matrix_bytes'],
                retained=ingress.resident_bytes+resident)*2+65536)
            remaining = {**budget,
                'maximum_matrix_bytes':budget['maximum_matrix_bytes']-ingress.resident_bytes-resident,
                'maximum_source_bytes':budget['maximum_source_bytes']-ingress.metrics['source_bytes']-source}
            require(all(v > 0 for v in remaining.values()), 'joint audit byte budget exceeded')
            owner = stack.enter_context(load_stock_ml_batch_inputs(manifest, residency='sequential', limits=remaining))
            state = _data(owner)['matrix_state']; states[owner_ref] = state; owners[owner_ref] = owner
            # Remaining is an ingress allowance. Subsequent leases enforce the
            # full joint budget with the other live owners/root as shared bytes.
            state.store.limits = dict(budget)
            _view_data(state.feature, check=False)['store'].limits = dict(budget)
            resident, source = _live(states)
            ingress.shared_bytes = resident; ingress.shared_source_bytes = source
            ingress.reserve(0)
            require(source+ingress.metrics['source_bytes'] <= budget['maximum_source_bytes'],
                'joint audit source byte budget exceeded')
        usage = {}
        admitted, records, epochs, manifests, labels = _admit_multi(
            {n:receipt['signal_inputs'][n] for n in root['signal_order']}, receipt['raw_label_input'],
            root['scope'], owners, receipt['prediction_owner_refs'], budget,
            _retained_bytes=ingress.resident_bytes, _retained_source_bytes=ingress.metrics['source_bytes'], _usage=usage)
        # A single completed-admission measurement bounds comparison workspace;
        # no historical graph is measured once per prediction item.
        graph = _size([admitted, records, epochs, manifests, labels], maximum=budget['maximum_matrix_bytes'],
            retained=ingress.resident_bytes+_live(states)[0])
        ingress.shared_bytes = _live(states)[0]+graph
        ingress.shared_source_bytes = _live(states)[1]+usage['label_source_bytes']
        ingress.reserve(graph+65536)
        expected = _root(receipt['signal_inputs'], receipt['raw_label_input'], root['scope'], admitted, records,
            manifests, receipt['prediction_owner_refs'], labels, budget)
        ingress.shared_bytes += _size(expected, maximum=ingress.maximum_matrix_bytes,
            retained=ingress.shared_bytes+ingress.resident_bytes+ingress.lease_bytes)
        ingress.reserve(0)
        for key in ('signal_refs', 'signal_metadata', 'raw_metadata', 'clock_floor'):
            require(expected[key] == root[key], 'audit original multi-owner projection mismatch')
        for key in ('source_records', 'source_closure', 'owner_manifests', 'label_manifest'):
            require(expected['admission_receipt'][key] == receipt[key], 'audit original owner closure mismatch')
        for day, descriptor in root['shards'].items():
            retained = ingress.shared_bytes
            saved, _ = _read_checked(Path(ref.uri).parent/descriptor['file'], descriptor['file_digest'],
                marks=marks, _budget=ingress)
            ingress.reserve(len(root['scope']['universe'])*(4096+4096*len(root['signal_order']))+65536)
            require(saved == _shard(admitted, root['scope'], day), 'audit original multi-owner date mismatch')
            saved = None; ingress.shared_bytes = retained
        _check_marks(epochs)
        ingress.validate_boundary()
    return ref

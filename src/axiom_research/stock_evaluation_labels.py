"""Independent saved evaluation Labels over the existing column/Core codec."""
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path

from .stock_artifacts import digest, file_digest, write_json
from .stock_fold_inputs import require
from .stock_compact_store import OwnedStore, limits as owner_limits, _size
from .stock_signal_evaluation_inputs import _scope
from .stock_target_spec import resolve_stock_label_spec

VERSION = 'stock_evaluation_label_inputs_v1'


def _identity(manifest):
    return digest({'contract_version': manifest['contract_version'],
        'definition': manifest['definition'], 'raw_parts': [
            {k: part[k] for k in ('file_digest', 'target_ref')} for part in manifest['raw_parts']]})


def _definition(snapshot, pit_policy, label_spec, scope):
    from .stock_compact_labels import _column_raw_implementation
    target = resolve_stock_label_spec(label_spec)
    require(type(snapshot) is str and snapshot not in ('', 'latest', 'current') and
            type(pit_policy) is str and bool(pit_policy), 'fixed evaluation Snapshot/PIT required')
    return {'snapshot': snapshot, 'pit_policy': pit_policy,
        'label_spec': target['label_spec'], 'label_spec_ref': target['label_definition_ref'],
        'scope': scope, 'implementation_ref': digest({
            'producer': inspect.getsource(build_stock_evaluation_label_inputs),
            'definition': inspect.getsource(_definition), 'identity': inspect.getsource(_identity),
            'raw_implementation_ref': _column_raw_implementation()})}


def _read_manifest(descriptor, store):
    from .stock_compact_store import reference
    require(type(descriptor) is dict and set(descriptor) == {'path', 'file_digest', 'label_ref'} and
        type(descriptor['path']) is str and Path(descriptor['path']).is_absolute() and
        reference(descriptor['file_digest']) and reference(descriptor['label_ref']),
        'fixed independent evaluation Label descriptor required')
    manifest = store.read_json(descriptor)
    require(type(manifest) is dict and set(manifest) == {'contract_version', 'definition', 'raw_parts', 'label_ref'} and
        manifest['contract_version'] == VERSION and manifest['label_ref'] == descriptor['label_ref'] == _identity(manifest),
        'independent evaluation Label identity/fields mismatch')
    definition = manifest['definition']
    require(set(definition) == {'snapshot', 'pit_policy', 'label_spec', 'label_spec_ref', 'scope', 'implementation_ref'} and
        reference(definition['implementation_ref']) and type(definition['snapshot']) is str and
        definition['snapshot'] not in ('', 'latest', 'current') and type(definition['pit_policy']) is str and
        bool(definition['pit_policy']), 'independent Label definition mismatch')
    target = resolve_stock_label_spec(definition['label_spec'])
    require(target['label_definition_ref'] == definition['label_spec_ref'] and
        target['label_spec'] == definition['label_spec'] and _scope(definition['scope']) == definition['scope'],
        'independent Label spec/scope mismatch')
    require(type(manifest['raw_parts']) is list and bool(manifest['raw_parts']), 'complete Raw parts required')
    # This detached manifest remains live when each part's backing is evicted.
    charge = _size(manifest, maximum=store.maximum_matrix_bytes,
                   retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)
    store.reserve(charge); store.shared_bytes += charge
    return manifest


def _parts(manifest, store):
    """Admit original typed targets and all source controls, one part at a time."""
    from .stock_compact_batch import read_target
    from .stock_signal_evaluation_compact import _header
    from .stock_column_inputs import validate_column_raw_binding
    definition = manifest['definition']; scope = definition['scope']
    common = {k: deepcopy(definition[k]) for k in ('snapshot', 'pit_policy')}
    common.update(calendar=scope['calendar'], universe=scope['universe'])
    common['target_spec'] = resolve_stock_label_spec(definition['label_spec'])
    seen = []
    for descriptor in manifest['raw_parts']:
        header, rows = read_target(store, descriptor)
        source = {'descriptor': descriptor, 'header': header}
        spec = _header(source, scope, common, store.hashes)
        raw = header['definition']
        require(header['contract_version'] == 'stock_compact_raw_v2' and
            raw['label_definition_ref'] == definition['label_spec_ref'] and
            raw['horizon_sessions'] == common['target_spec']['horizon_sessions'] and
            raw['cutoff'] == scope['evaluation_cutoff'], 'independent Label target definition mismatch')
        validate_column_raw_binding(raw, common, scope['evaluation_cutoff'])
        days = raw['sessions']
        require(not seen or seen[-1] < days[0], 'independent Label parts overlap or are unordered')
        seen.extend(days)
        yield descriptor, header, rows, spec
        rows = header = source = None
        store.release_payloads()
    require(seen == scope['sessions'], 'independent Label grid coverage mismatch')
    store.check()


def load_stock_evaluation_label_inputs(descriptor, *, limits=None):
    """Cold byte/source validation only; return detached immutable controls."""
    with OwnedStore(owner_limits(limits)) as store:
        manifest = _read_manifest(descriptor, store)
        for _ in _parts(manifest, store):
            pass
        return deepcopy(manifest)


def build_stock_evaluation_label_inputs(data, *, snapshot, pit_policy, label_spec, scope,
    column_source, destination, limits):
    """Save full-grid Raw outcomes, with no Feature, normalization or training.

    The caller owns its existing public ColumnSource. A matching saved manifest
    is validated and reused before any selection. Invalid/missing endpoints
    remain in the original typed target with their masks/reasons/clocks.
    """
    from .stock_compact_labels import _query, _raw, _working, _sync_feature_charge
    from .stock_column_inputs import ColumnPriceDomain
    scope = _scope(scope); definition = _definition(snapshot, pit_policy, label_spec, scope)
    budget = owner_limits(limits); root = Path(destination).resolve()/digest(definition)[7:]
    path = root/'manifest.json'
    if path.exists():
        manifest = OwnedStore(budget)
        try:
            saved = manifest.read_json({'path': str(path)})
            require(saved['definition'] == definition, 'cached evaluation Label definition mismatch')
            descriptor = {'path': str(path), 'file_digest': manifest.hashes[str(path)], 'label_ref': saved['label_ref']}
            _read_manifest(descriptor, manifest)
            for _ in _parts(saved, manifest):
                pass
            return descriptor
        finally:
            manifest.close()
    spec = {k: deepcopy(definition[k]) for k in ('snapshot', 'pit_policy')}
    spec.update(calendar=scope['calendar'], universe=scope['universe'], feature_sessions=scope['sessions'],
                target_spec=resolve_stock_label_spec(definition['label_spec']))
    with OwnedStore(budget) as store, OwnedStore(budget) as empty_feature_store:
        controls = _size([spec, definition]); store.reserve(controls)
        stats = {'_limits': budget, '_store': store, '_feature_store': empty_feature_store,
            '_caller_bytes': controls, '_caller_source_bytes': 0, '_feature_bytes': controls,
            '_retained_raw_bytes': 0, '_price_view_bytes': 0, '_column_targets': None,
            'maximum_working_bytes': controls, 'raw_cache_hits': 0, 'raw_operator_calls': 0,
            'data_read_calls': 0}
        raw_parts = []
        # This is the existing fixed Raw block codec; each panel is released
        # before the next public select, never a second Feature executor.
        for offset in range(0, len(scope['sessions']), 64):
            days = scope['sessions'][offset:offset+64]
            query, anchor = _query(spec, scope['evaluation_cutoff'], days)
            _working(stats, len(query.sessions)*len(query.symbols)*160+65536)
            prices = factors = adjusted = domain = rows = None
            try:
                prices = column_source.select(query=query)
                factors = column_source.select(query=replace(query, domain='adjustment_factors', fields=('factor',)))
                adjusted = column_source.adjust(prices, factors, fields=('open', 'close'),
                    anchor_session=anchor, decision_session=anchor, factor_field='factor')
                domain = ColumnPriceDomain(adjusted, spec=spec,
                    query=replace(query, price_basis='common_anchor_adjusted_v1', adjustment_anchor=anchor),
                    anchor=anchor, input_queries={'price': prices.query_binding, 'factor': factors.query_binding}, metrics=stats)
                stats['_price_view_bytes'] = sum(a.nbytes for panel in domain.panels.values()
                    for a in panel.values() if hasattr(a, 'nbytes')) + sum(a.nbytes for a in domain.lineage.values()) + _size([domain.source, domain.positions])
                _sync_feature_charge(stats); _working(stats, 0)
                descriptor, rows = _raw(spec, scope['evaluation_cutoff'], days, root, stats, domain)
                raw_parts.append(descriptor)
            finally:
                rows = None
                if domain is not None:
                    domain.close()
                for selection in (adjusted, factors, prices):
                    if selection is not None:
                        selection.close()
                store.release_payloads()
            stats['_caller_bytes'] = controls + _size(raw_parts)
            _working(stats, 0)
        manifest = {'contract_version': VERSION, 'definition': definition, 'raw_parts': raw_parts}
        manifest['label_ref'] = _identity(manifest)
        root.mkdir(parents=True, exist_ok=True)
        from tempfile import TemporaryDirectory
        import os
        with TemporaryDirectory(prefix='.evaluation-label-', dir=root) as temporary:
            stage = Path(temporary)/'manifest.json'; write_json(stage, manifest)
            descriptor = {'path': str(stage), 'file_digest': file_digest(stage), 'label_ref': manifest['label_ref']}
            load_stock_evaluation_label_inputs(descriptor, limits=budget)
            try:
                os.link(stage, path)
            except FileExistsError:
                require(file_digest(path) == descriptor['file_digest'], 'conflicting evaluation Label publication')
        final = {'path': str(path), 'file_digest': descriptor['file_digest'], 'label_ref': manifest['label_ref']}
        # Retain the verified byte digest across publication; a changed writer
        # output must reject rather than become the descriptor's new baseline.
        load_stock_evaluation_label_inputs(final, limits=budget)
        return final

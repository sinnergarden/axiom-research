"""Bounded opt-in Feature input preparation through the existing Data/Core path.

Qlib supplies an immutable native projection, not revision selection. Every
output still reads its original history at its declared cutoff and uses Core
for the complete cross-section. The public Core batch receives independent
original daily views; shared-panel classification remains diagnostic.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import chain
from pathlib import Path
import sys
import time

from .stock_artifacts import digest


FIELDS = ('open', 'high', 'low', 'close', 'amount_cny', 'factor')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _positive(value, name):
    _require(type(value) is int and value > 0, 'positive integer '+name+' required')


def _owned_bytes(value, seen=None):
    """Account this bounded Python graph; this is not process-tree RSS."""
    seen = set() if seen is None else seen
    amount = 0
    # Depth-first iterators preserve key/value order and shared-object accounting
    # without recursive calls or materializing a wide child list.
    stack = [iter((value,))]
    while stack:
        for item in stack[-1]:
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
            amount += sys.getsizeof(item)
            if isinstance(item, dict):
                stack.append(chain.from_iterable(item.items()))
                break
            if isinstance(item, (list, tuple, set, frozenset)):
                stack.append(iter(item))
                break
            if type(getattr(item, 'payload', None)) is str:
                # Keep the existing conservative Document dictionary charge,
                # including when that dictionary is reachable separately.
                amount += _owned_bytes(item.payload, seen)
                amount += sys.getsizeof(getattr(item, '__dict__', {}))
        else:
            stack.pop()
    return amount


def _header_bytes(*roots):
    """Visible field lengths only; never descend into a provenance graph."""
    headers = list(roots)
    for value in roots:
        if type(value) is dict:
            headers.extend(value.values())
            for child in value.values():
                if type(child) is dict:
                    headers.extend(child.values())
    return sum(sys.getsizeof(value) for value in headers)


def _batch_bytes(batch, *, stats=None):
    """Block charge estimate, not a recursive proof of every provenance byte.

    Data owns source admission. Keep frame dimensions and visible field lengths
    here; arbitrary nested provenance is covered by the declared estimate/RSS
    policy, rather than rewalking its graph at each private transformation.
    """
    tick = time.perf_counter_ns()
    amount = (int(batch.frame.memory_usage(deep=True).sum())+
              len(batch.frame)*max(1, len(batch.frame.columns))*512+
              _header_bytes(batch.field_meta, batch.context))
    if stats is not None:
        stats['reader_batch_charge_estimates'] = stats.get('reader_batch_charge_estimates',0)+1
        stats['reader_batch_charge_estimate_ns'] = stats.get('reader_batch_charge_estimate_ns',0)+time.perf_counter_ns()-tick
    return amount


def _caller_bytes(getter):
    if getter is None:
        return 0
    _require(callable(getter), 'caller_retained_bytes must be callable')
    value = getter()
    _require(type(value) is int and value >= 0, 'caller_retained_bytes must return a nonnegative integer')
    return value


def _guard_resident(owned, *, maximum_resident_bytes, stats, caller_retained_bytes, reason, owner_limit=None):
    """Check a block working-set estimate; neither counter measures RSS."""
    owned += 64*1024  # bounded scalar diagnostics allowance, no graph scan
    caller = _caller_bytes(caller_retained_bytes)
    combined = owned+caller
    stats['working_graph_peak_bytes'] = max(stats.get('working_graph_peak_bytes', 0), owned)
    stats['combined_working_graph_peak_bytes'] = max(stats.get('combined_working_graph_peak_bytes', 0), combined)
    _require(owned <= (maximum_resident_bytes if owner_limit is None else owner_limit), reason)
    _require(combined <= maximum_resident_bytes, 'combined producer/caller resident budget exceeded: '+reason)


def prepare_matrix_qlib(data, *, config, destination):
    """Export and activate the original view without loading its native panel."""
    from axiom_data import QuerySpec
    from .qlib_adapter import QlibView
    symbols = tuple(config['symbols'])
    sessions = tuple(config['read_sessions'])
    price = QuerySpec('market_daily', FIELDS[:-1], symbols, sessions,
                     config['pit_policy'], config['cutoff_by_session'])
    factor = replace(price, domain='adjustment_factors', fields=('factor',))
    member = replace(price, domain='universe_membership', fields=('is_member',),
                     universe_id=config['universe_id'])
    begin = time.perf_counter()
    # Keep the original export namespace/query construction. The new producer
    # implementation enters the opt-in Feature definition independently.
    path = Path(destination)/'qlib'/digest({'snapshot': config['snapshot'],
        'prices': {k: str(v) for k, v in vars(price).items()}, 'scope_ref': config['scope_ref']})[7:]
    data.export_qlib(snapshot=config['snapshot'], queries=(price, factor), destination=path,
                     universe_query=member, universe_name=config['universe_id'])
    view = QlibView(path).activate()
    return {'view': view, 'view_reference': view.reference, 'price_query': price,
            'factor_query': factor, 'member_query': member, 'qlib_path': path,
            'seconds': time.perf_counter()-begin}


class _NativeWindow:
    """One bounded keyed native DataFrame; never a whole-period values dict."""
    def __init__(self, native, *, sessions, symbols, instrument_map):
        import pandas as pd
        _require(isinstance(native, pd.DataFrame) and isinstance(native.index, pd.MultiIndex)
                 and native.index.nlevels == 2, 'keyed Qlib native window required')
        _require(list(native.columns) == ['$'+field for field in FIELDS],
                 'exact Qlib native window fields required')
        reverse = {value: key for key, value in instrument_map.items()}
        _require(len(reverse) == len(instrument_map), 'duplicate Qlib instrument mapping')
        keys = []
        for instrument, day in native.index:
            _require(instrument in reverse, 'Qlib window contains unknown instrument')
            timestamp = pd.Timestamp(day)
            _require(not pd.isna(timestamp) and timestamp.time().isoformat() == '00:00:00',
                     'Qlib window requires session dates')
            keys.append((reverse[instrument], timestamp.date().isoformat()))
        expected = {(security, session) for session in sessions for security in symbols}
        _require(len(keys) == len(set(keys)) and set(keys) == expected,
                 'complete unique Qlib native window grid required')
        self.frame = native.copy(deep=False)
        self.frame.index = pd.MultiIndex.from_tuples(keys, names=['security_id', 'session'])
        self.bytes = int(self.frame.memory_usage(deep=True).sum())

    def project(self, batch, *, view_ref, stats, maximum_resident_bytes, retained_bytes=0,
                caller_retained_bytes=None, _with_wire=False):
        """Round actual Reader values exactly as the native float32 format.

        Equal native values can be reused. Revision/missingness differences
        use the current Reader value, with its unaltered field provenance.
        This operation performs no Feature or adjustment mathematics.
        """
        import numpy as np
        import pandas as pd
        from axiom_data import DataBatch
        # Reject a large coverage/source graph before to_json duplicates it.
        reserve = self.bytes+retained_bytes+2*_batch_bytes(batch,stats=stats)+len(batch.frame)*len(FIELDS)*32
        # A private projected wire adds one shallow row dictionary per key and
        # new numeric scalars. Charge its list/rows before either wire exists.
        if _with_wire:
            reserve += sys.getsizeof([])+len(batch.frame)*(512+128*len(batch.frame.columns))
        _guard_resident(reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, reason='Reader projection working set exceeds resident budget')
        begin = time.perf_counter_ns()
        wire = batch.to_json()
        query = wire['context']['query']
        fields = query['fields']
        keys = [(row['security_id'], row['session']) for row in wire['records']]
        expected = {(security, day) for day in query['sessions'] for security in query['symbols']}
        _require(len(keys) == len(set(keys)) and set(keys) == expected,
                 'complete unique Reader projection grid required')
        _require(set(fields) <= set(FIELDS) and set(keys) <= set(self.frame.index),
                 'Reader projection outside native window')
        native = self.frame.reindex(pd.MultiIndex.from_tuples(keys))
        frame = batch.frame.copy()
        projected_records = [dict(row) for row in wire['records']] if _with_wire else None
        for field in fields:
            missing = np.asarray([row[field] is None for row in wire['records']], dtype=bool)
            original = np.asarray([row[field] for row in wire['records']], dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                projected = original.astype(np.float32).astype(np.float64)
            _require(bool(np.all(np.isfinite(projected[~missing]))),
                     'nonfinite Reader float32 projection')
            values = native['$'+field].to_numpy(dtype=np.float64)
            native_missing = np.isnan(values)
            _require(bool(np.all(np.isfinite(values[~native_missing]))), 'nonfinite Qlib native projection')
            matches = (missing & native_missing) | (~missing & ~native_missing & (projected == values))
            selected = np.where(matches, values, projected)
            selected[missing] = np.nan
            stats['native_value_reuses'] += int(np.count_nonzero(matches & ~missing))
            stats['reader_projection_fallback_cells'] += int(np.count_nonzero(~matches))
            stats['reader_projection_fallback_null_cells'] += int(np.count_nonzero(~matches & missing))
            dtype = frame[field].dtype
            if pd.api.types.is_integer_dtype(dtype):
                bounds = np.iinfo(getattr(dtype, 'numpy_dtype', dtype))
                _require(all(np.isfinite(value) and bounds.min <= int(value) <= bounds.max and
                             float(int(value)) == value for value in selected[~missing]),
                         'lossy integer Qlib projection assignment')
                frame[field] = pd.array(selected, dtype=dtype)
            else:
                frame[field] = selected
            if projected_records is not None:
                convert = int if pd.api.types.is_integer_dtype(dtype) else float
                for row, value, absent in zip(projected_records, selected, missing):
                    row[field] = None if absent else convert(value)
        context = {**wire['context'], 'numeric_projection': {
            'contract_version': 'research_qlib_native_projection_v2', 'view_ref': view_ref,
            'reader_batch_ref': digest(wire),
            'revision_admission': 'actual_reader_float32_with_native_value_reuse',
            'dtype': 'float32_values_promoted_to_float64'}}
        # to_json already owns canonical metadata. Keep this selected snapshot
        # isolated from later mutation of the original Reader object.
        result = DataBatch(frame, wire['field_meta'], context)
        stats['native_projection_ns'] = stats.get('native_projection_ns', 0)+time.perf_counter_ns()-begin
        if _with_wire:
            return result, {'records': projected_records, 'field_meta': wire['field_meta'], 'context': context}
        return result


def _view_signature(plan, facts, context, *, calendar=None, _wires=None):
    """Compact exact input compatibility evidence, without a second executor."""
    p, f, c = (plan.to_dict(), facts.to_dict(), context.to_dict()) if _wires is None else _wires
    bindings = {source['id']: source for source in p['sources']}
    calendar = c['sessions'] if calendar is None else list(calendar)
    _require(calendar == sorted(set(calendar)) and set(c['sessions']) <= set(calendar),
             'exact shared calendar scope required')
    _require(c['sessions'] == calendar[calendar.index(c['sessions'][0]):calendar.index(c['sessions'][-1])+1],
             'Feature history must preserve calendar gaps')
    return {'recipe': digest({key: p[key] for key in ('abi', 'semantics', 'recipe_ref',
                'input_schema', 'event_schema', 'observation_domain', 'history_policy', 'nodes', 'outputs')}),
        'calendar_ref': digest(calendar),
        'sessions': c['sessions'], 'cutoffs': c['cutoffs'],
        'output_keys': c['output_keys'],
        'facts': {(row['security_id'], row['session']): {
            'values': digest(row['values']), 'availability': digest(row['availability']),
            'reasons': digest(row['missing_reasons']),
            'sources': digest([[bindings[source] for source in sources] for sources in row['sources']])}
            for row in f['rows']},
        'reference': {(row['security_id'], row['session']): digest({**row,
            'binding': bindings[row['source']]}) for row in c['reference']}}


def classify_shared_feature_views(left, right):
    """Classify two compact signatures; no computation or inferred grouping.

    A compatible answer describes existing Core's shared-panel input condition.
    It does not authorize aliasing a grouped Frame to either original view.
    """
    reasons = []
    if left['recipe'] != right['recipe']:
        reasons.append('RECIPE_SCHEMA_CONFLICT')
    if left['calendar_ref'] != right['calendar_ref']:
        reasons.append('CALENDAR_SCOPE_CONFLICT')
    overlap = set(left['facts']) & set(right['facts'])
    for name, reason in [('values', 'FACT_VALUE_CONFLICT'), ('availability', 'FACT_CLOCK_CONFLICT'),
                         ('reasons', 'FACT_MISSING_REASON_CONFLICT'), ('sources', 'FACT_SOURCE_CONFLICT')]:
        if any(left['facts'][key][name] != right['facts'][key][name] for key in overlap):
            reasons.append(reason)
    if any(left['reference'][key] != right['reference'][key] for key in overlap):
        reasons.append('REFERENCE_CONFLICT')
    sessions = sorted(set(left['sessions']) | set(right['sessions']))
    cutoffs = {session: min(view['cutoffs'][session] for view in (left, right)
                           if session in view['cutoffs']) for session in sessions}
    if [cutoffs[session] for session in sessions] != sorted(cutoffs.values()):
        reasons.append('NONMONOTONIC_SHARED_CUTOFF')
    for view in (left, right):
        if any(cutoffs[session] != view['cutoffs'][session] for _, session in view['output_keys']):
            reasons.append('OUTPUT_CUTOFF_CONFLICT')
            break
    return {'status': 'REQUIRES_VIEW_FALLBACK' if reasons else 'SHARED_PANEL_COMPATIBLE',
            'reasons': reasons}


_MEMORY_POINTWISE_OPS = frozenset(('constant', 'identity', 'add', 'sub', 'mul',
    'divide', 'abs', 'log1p', 'gt', 'lt', 'eq', 'and', 'or', 'not', 'where',
    'to_float', 'is_missing', 'fill', 'clip', 'calendar_age'))
_MEMORY_CS_OPS = frozenset(('cs_rank', 'cs_zscore', 'cs_winsorize'))


def _core_scope_counts(plan, history_count, symbol_count, *, output_position=None):
    """Count cells from the admitted public DAG, without computing any values.

    In this producer every output day requests the complete security cohort.
    Therefore each node's temporal scope is uniform across securities; CS needs
    the full cohort at its requested dates, including excluded members. The
    temporal rules mirror Core 60df1fc's dependency admission walk. Unsupported
    operations fail closed, rather than silently omitting their memory.
    """
    _positive(history_count, 'memory history_count')
    _positive(symbol_count, 'memory symbol_count')
    _require(not plan['event_schema'] and plan['observation_domain'] == 'sessions',
             'memory scope requires original session-only Feature views')
    position = history_count-1 if output_position is None else output_position
    _require(type(position) is int and 0 <= position < history_count, 'covered memory output position required')
    needed = {column['name']: set() for column in plan['input_schema']}
    needed.update({node['name']: set() for node in plan['nodes']})
    for output in plan['outputs']:
        needed[output['node']].add(position)
    for node in reversed(plan['nodes']):
        op, params = node['op'], node['params']
        _require(op in _MEMORY_POINTWISE_OPS | _MEMORY_CS_OPS | {'shift', 'pct_change', 'rolling'},
                 'unsupported Core memory scope operator: '+op)
        positions = needed[node['name']]
        dependencies = set()
        for index in positions:
            if op in ('shift', 'pct_change'):
                if index >= params['periods']:
                    dependencies.add(index-params['periods'])
                if op == 'pct_change':
                    dependencies.add(index)
            elif op == 'rolling':
                end = index+int(params['inclusive_current'])
                dependencies.update(range(max(0, end-params['window']), end))
            else:
                dependencies.add(index)
        for parent in node['inputs']:
            needed[parent].update(dependencies)
    nodes = {node['name']: len(needed[node['name']])*symbol_count for node in plan['nodes']}
    inputs = {column['name']: len(needed[column['name']])*symbol_count for column in plan['input_schema']}
    return {'history_keys': history_count*symbol_count, 'input_cells': history_count*symbol_count*len(inputs),
            'node_cells': sum(nodes.values()), 'nodes': nodes,
            'dependency_entries': sum(nodes.values())+sum(inputs.values()),
            'symbol_count': symbol_count, 'history_count': history_count}


def _core_memory_bounds(plan, dimensions, *, snapshots, input_sources, reference_sources):
    """Charge snapshots, scope maps, cells/refs, CS scratch and Frame copies.

    The 2048-byte cell allowance includes Python Cell/dict/list/tuple backing,
    values, clocks, reasons and mapping workspace. Its complete possible source
    strings are additionally charged per cell. Parser and dependency/index
    maps, full-union reduction/reference scratch and six output copies remain
    separate. Core's call-local memo is reserved by the caller independently.
    """
    sources = {name: set(value) for name, value in input_sources.items()}
    for node in plan['nodes']:
        possible = set().union(*(sources[parent] for parent in node['inputs']))
        if node['op'] in _MEMORY_CS_OPS:
            possible.update(reference_sources)
        sources[node['name']] = possible

    def source_bytes(name):
        return sum(sys.getsizeof(value)+32 for value in sources[name])

    cells = dimensions['history_keys']*sum(2048+source_bytes(column['name']) for column in plan['input_schema'])
    cells += sum(dimensions['nodes'][node['name']]*(2048+source_bytes(node['name'])) for node in plan['nodes'])
    maps = dimensions['history_keys']*512+dimensions['dependency_entries']*192
    scratch = max(dimensions['symbol_count'], dimensions['history_count'])*(
        4096+max((source_bytes(node['name']) for node in plan['nodes']), default=0))
    source_sets = 2*(sum(512+len(values)*128 for values in sources.values())+
                     sum(source_bytes(name) for name in sources)+len(reference_sources)*256)
    workspace = 3*snapshots+cells+maps+scratch+source_sets
    output = 6*(len(plan['sources'])*2048+len(plan['outputs'])*512+dimensions['symbol_count']*(256+
        sum(1024+source_bytes(item['node']) for item in plan['outputs'])))
    return workspace, output


def _core_budget_plan(catalog, selection):
    """Compile only the original selected DAG on a tiny neutral plan header."""
    from axiom_engine.core import ABI, SEMANTICS, FeaturePlan
    from .feature_catalog import build_feature_plan
    binding = digest('memory-shape-only')
    fields = FIELDS[:-1]
    header = {'abi': ABI, 'semantics': SEMANTICS, 'recipe_ref': binding,
        'calendar_ref': binding, 'reference_ref': binding,
        'reference_members': {'2000-01-03': {'memory-shape': None}},
        'input_schema': [{'name': field, 'dtype': 'float64', 'unit': 'CNY' if field == 'amount_cny' else 'CNY/share',
                         'stage': 'fact', 'missing': 'preserve'} for field in fields],
        'event_schema': {}, 'sources': [{'id': digest(['memory-shape', field]), 'data_ref': binding,
            'view_ref': binding, 'revision_policy': 'memory_shape', 'qualification': 'synthetic',
            'availability_basis': 'memory_shape'} for field in (*fields, 'is_member')],
        'observation_domain': 'sessions', 'history_policy': 'partial', 'nodes': [], 'outputs': [], 'obligations': []}
    return build_feature_plan(FeaturePlan.from_dict(header), selection, catalog=catalog, normalized=True).to_dict()


def _core_preflight_bounds(plan, history_count, symbol_count):
    """Initial shape estimate; actual source variants/graphs tighten it later."""
    dimensions = _core_scope_counts(plan, history_count, symbol_count)
    input_sources = {column['name']: {digest(['memory-shape', column['name']])} for column in plan['input_schema']}
    snapshots = _owned_bytes(plan)+dimensions['history_keys']*6144
    workspace, output = _core_memory_bounds(plan, dimensions, snapshots=snapshots,
        input_sources=input_sources, reference_sources={digest(['memory-shape', 'is_member'])})
    return workspace, output, dimensions


def _core_view_bounds(plan, facts, context, *, snapshots):
    """Reserve one actual original view's precise dependency dimensions."""
    symbols = {key[0] for key in context['history_keys']}
    dates = context['sessions']
    days = {key[1] for key in context['output_keys']}
    _require(len(days) == 1 and len(context['history_keys']) == len(symbols)*len(dates) and
             {tuple(key) for key in context['output_keys']} == {(symbol, next(iter(days))) for symbol in symbols},
             'memory scope requires complete original daily cohort')
    dimensions = _core_scope_counts(plan, len(dates), len(symbols), output_position=dates.index(next(iter(days))))
    input_sources = {column['name']: set() for column in plan['input_schema']}
    for row in facts['rows']:
        for column, bindings in zip(plan['input_schema'], row['sources']):
            input_sources[column['name']].update(bindings)
    return _core_memory_bounds(plan, dimensions, snapshots=snapshots,
        input_sources=input_sources, reference_sources={row['source'] for row in context['reference']})


def _measure_owned_bytes(value, *, stats, kind):
    tick = time.perf_counter_ns()
    amount = _owned_bytes(value)
    stats[kind+'_size_measurements'] = stats.get(kind+'_size_measurements', 0)+1
    stats[kind+'_size_measurement_ns'] = stats.get(kind+'_size_measurement_ns', 0)+time.perf_counter_ns()-tick
    return amount


def _seal_pending_view(request, evidence, members, workspace, output_reserve, *, stats, owned_estimate=None):
    # These private objects have no caller alias. Documents are immutable and
    # evidence/members remain read-only until this entry is removed at delivery.
    # Cache only this owned lifetime, never an id/ref for an external graph.
    view = (request, evidence, members, workspace, output_reserve)
    amount = (_measure_owned_bytes(view, stats=stats, kind='pending_view')
              if owned_estimate is None else owned_estimate)
    if owned_estimate is not None:
        stats['pending_view_charge_estimates'] = stats.get('pending_view_charge_estimates', 0)+1
    # Cover the extra tuple slot, cached integer and accounting scalar headers.
    # Separate views are measured separately: shared children are overcounted.
    return (*view, amount+512)


def _pending_owned_bytes(pending):
    return sys.getsizeof(pending)+sum(view[5] for view in pending if view is not None)


def _pending_core_bytes(pending):
    views = [view for view in pending if view is not None]
    return (_pending_owned_bytes(pending)+max((view[3] for view in views), default=0)+
            sum(view[4] for view in views)+sys.getsizeof(views))


def _record_core_batch_stats(stats, actual):
    stats['core_batch_views'] += actual['views']
    for name in ('key_attempts', 'keys_built', 'key_charge_bytes', 'key_build_ns',
                 'numeric_compute_ns', 'reuse_overhead_ns', 'budget_fallbacks',
                 'evictions', 'numeric_ids_built', 'total_ns'):
        key = 'core_batch_'+name
        stats[key] = stats.get(key, 0)+actual[name]
    stats['core_batch_peak_reuse_bytes'] = max(stats.get('core_batch_peak_reuse_bytes', 0),
                                             actual['peak_reuse_bytes'])
    stats['core_batch_retained_reuse_bytes'] = actual['retained_reuse_bytes']
    for operation, counts in actual['helpers'].items():
        combined = stats['core_batch_helpers'].setdefault(operation, {})
        for name, value in counts.items():
            combined[name] = combined.get(name, 0)+value
    for operation, reason in actual['disabled_ops'].items():
        counts = stats['core_batch_disabled_ops'].setdefault(operation, {})
        counts[reason] = counts.get(reason, 0)+1
    stats['core_batch_total_seconds'] = stats['core_batch_total_ns']/1_000_000_000


def _iter_core_feature_batch(pending, *, resident_base, maximum_resident_bytes,
                             reuse_budget_bytes, stats, caller_retained_bytes,
                             progress, completed, total, begin):
    from axiom_engine.core import execute_feature_plan_batch
    requests = tuple(view[0] for view in pending)
    # A fresh caller check immediately precedes every actual public execution.
    # Documents/request tuples are already covered by the sealed view charges.
    reserve = resident_base+_pending_core_bytes(pending)+reuse_budget_bytes+sys.getsizeof(requests)
    _guard_resident(reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
        caller_retained_bytes=caller_retained_bytes, reason='Core batch snapshots/output working set exceeds resident budget')
    stats['core_calls'] += 1
    stats['feature_core_calls'] += 1
    stats['core_batch_calls'] += 1
    stats['actual_core_multi_view_batches'] += int(len(requests) > 1)
    stats['core_batch_peak_views'] = max(stats['core_batch_peak_views'], len(requests))
    tick = time.perf_counter_ns()
    result = execute_feature_plan_batch(requests, reuse_budget_bytes=reuse_budget_bytes)
    stats['core_batch_public_wall_ns'] += time.perf_counter_ns()-tick
    _require(set(result) == {'frames', 'stats'} and type(result['frames']) is tuple and
             len(result['frames']) == len(requests) and result['stats']['views'] == len(requests),
             'complete original Core batch Frames required')
    _record_core_batch_stats(stats, result['stats'])
    frames = list(result['frames'])
    del result, requests
    frame_charges = [_measure_owned_bytes(frame, stats=stats, kind='core_frame') for frame in frames]
    remaining_bytes = _pending_owned_bytes(pending)+sys.getsizeof(frames)+sum(frame_charges)
    charge_controls = _owned_bytes(frame_charges)+512
    stats['core_daily_frames'] += len(frames)
    stats['core_single_output_groups'] += len(frames)
    try:
        for index in range(len(frames)):
            request, evidence, members, _, output_reserve, view_charge = pending[index]
            plan, facts, context = request
            frame = frames[index]
            # Retain all not-yet-yielded views/Frames in the accounting. Parsing
            # a Frame and building row/proof copies also needs its own reserve.
            live = resident_base+remaining_bytes+charge_controls+output_reserve
            _guard_resident(live, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature output working graph exceeds resident budget')
            frame_wire = frame.to_dict()
            day = evidence['session']
            frame_ref, plan_ref = frame.identity, plan.identity
            rows = [{'security_id': row['security_id'], 'session': day, 'values': row['values'],
                'availability': row['availability'], 'validity': row['valid'], 'reasons': row['reasons'],
                'member': members[row['security_id']], 'knowledge_cutoff': evidence['cutoffs'][day],
                'source_refs': [frame_ref, plan_ref]} for row in frame_wire['rows']]
            evidence['core_frame_ref'] = frame_ref
            pending[index] = frames[index] = None
            remaining_bytes -= view_charge+frame_charges[index]
            del request, plan, facts, context, frame, frame_wire, members
            completed += 1
            if progress is not None:
                progress({'stage': 'matrix_features', 'completed': completed, 'total': total,
                          'session': day, 'seconds': time.perf_counter()-begin})
            # Keep the reserved row/proof allowance through delivery. Private
            # assembly changes no business checks and needs no graph rewalk.
            live = resident_base+remaining_bytes+charge_controls+view_charge+output_reserve
            _guard_resident(live, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature yield working graph exceeds resident budget')
            stats['yield_live_bytes'] = live
            try:
                yield rows, evidence
            finally:
                stats['yield_live_bytes'] = 0
            del rows, evidence
    finally:
        pending.clear()
        frames.clear()
    return completed


def iter_matrix_feature_days(data, *, config, catalog, chosen, qlib_inputs,
                             history_sessions=21, output_block_sessions=64,
                             maximum_resident_bytes, progress=None, stats=None,
                             caller_retained_bytes=None, reuse_budget_bytes=None):
    """Yield original daily rows/proofs from bounded public multi-view calls.

    Reuse defaults to one eighth of the existing resident budget. This internal
    option adds no public storage definition fields. Its cache allowance is
    separate from pending original documents, Core workspace and returned Frames.
    """
    from .data_adapter import _adapt_decision_wires
    from .feature_catalog import build_feature_plan
    from .stock_ml import _adjust_feature_wire
    for value, name in [(history_sessions, 'history_sessions'), (output_block_sessions, 'output_block_sessions'),
                        (maximum_resident_bytes, 'maximum_resident_bytes')]:
        _positive(value, name)
    _require(history_sessions == max(entry['lookback'] for entry in chosen),
             'Feature history differs from selected catalog lookback')
    if reuse_budget_bytes is None:
        reuse_budget_bytes = maximum_resident_bytes//8
    _require(type(reuse_budget_bytes) is int and reuse_budget_bytes >= 0,
             'nonnegative integer reuse_budget_bytes required')
    _require(reuse_budget_bytes <= maximum_resident_bytes, 'reuse budget exceeds resident budget')
    stats = {} if stats is None else stats
    for name in ('data_read_calls', 'core_calls', 'feature_core_calls', 'native_window_reads',
                 'native_window_peak_rows', 'native_window_peak_sessions', 'native_window_peak_bytes',
                 'working_graph_peak_bytes', 'combined_working_graph_peak_bytes', 'yield_live_bytes',
                 'native_value_reuses', 'reader_projection_fallback_cells',
                 'reader_projection_fallback_null_cells', 'core_single_output_groups',
                 'actual_core_multi_output_groups', 'actual_core_multi_view_batches',
                 'core_batch_calls', 'core_batch_views', 'core_batch_peak_views', 'core_daily_frames',
                 'core_batch_public_wall_ns', 'native_window_read_ns', 'native_projection_ns',
                 'data_read_ns', 'feature_adaptation_ns', 'pending_view_peak_bytes', 'batch_budget_flushes'):
        stats.setdefault(name, 0)
    stats.setdefault('core_batch_helpers', {})
    stats.setdefault('core_batch_disabled_ops', {})
    stats.setdefault('classifier_status_counts', {})
    stats.setdefault('classifier_reason_counts', {})
    stats['group_execution_status'] = 'ORIGINAL_DAILY_VIEWS_PUBLIC_CORE_BATCH'
    stats['core_call_counter_basis'] = 'public_execute_feature_plan_batch_invocations'
    stats['core_reuse_budget_bytes'] = reuse_budget_bytes
    stats['resource_accounting_basis'] = 'block_owned_estimate_not_process_tree_rss'
    _guard_resident(0, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                    caller_retained_bytes=caller_retained_bytes, reason='one native Feature window exceeds resident budget')
    symbols = tuple(config['symbols'])
    sessions = tuple(config['read_sessions'])
    outputs = list(config['feature_sessions'])
    positions = {day: index for index, day in enumerate(sessions)}
    _require(outputs == sorted(set(outputs)) and set(outputs) <= set(sessions),
             'ordered covered matrix Feature outputs required')
    _require(all(positions[day]+1 >= history_sessions for day in outputs), 'Feature lookback incomplete')
    begin = time.perf_counter()
    begin_ns = time.perf_counter_ns()
    start = 0
    completed = 0
    base = _owned_bytes([config, chosen, catalog.payload, qlib_inputs['view_reference']])
    # Before reading, reserve metadata/wires and possible Core cells as well as
    # native numbers. Later checks use exact Plan node/source dimensions and
    # call-local block estimates, not recursive temporary-graph measurements.
    daily_keys = history_sessions*len(symbols)
    read_estimate = daily_keys*(8192+1024*len(FIELDS))
    pending_estimate = daily_keys*(2048+256*len(FIELDS))
    _guard_resident(base+reuse_budget_bytes+12*_owned_bytes(chosen)+3*_owned_bytes(catalog.payload),
        maximum_resident_bytes=maximum_resident_bytes, stats=stats,
        caller_retained_bytes=caller_retained_bytes, reason='Feature scope plan compilation exceeds resident budget')
    budget_plan = _core_budget_plan(catalog, config['feature_selection'])
    core_estimate, output_estimate, dimensions = _core_preflight_bounds(budget_plan, history_sessions, len(symbols))
    stats['core_preflight_input_cells'] = dimensions['input_cells']
    stats['core_preflight_node_cells'] = dimensions['node_cells']
    stats['core_preflight_workspace_bytes'] = core_estimate
    stats['core_preflight_output_bytes'] = output_estimate
    base += _owned_bytes(budget_plan)
    del dimensions
    while start < len(outputs):
        count = min(output_block_sessions, len(outputs)-start)
        while True:
            block = outputs[start:start+count]
            history = sessions[positions[block[0]]-history_sessions+1:positions[block[-1]]+1]
            # Reserve room for Reader, derived wires and Core snapshots too.
            estimate = len(history)*len(symbols)*(512+32*len(FIELDS))
            combined = base+estimate+reuse_budget_bytes+read_estimate+core_estimate+count*(pending_estimate+output_estimate)
            if (estimate <= maximum_resident_bytes//3 and
                    combined+_caller_bytes(caller_retained_bytes) <= maximum_resident_bytes) or count == 1:
                break
            count = max(1, count//2)
        _guard_resident(estimate, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, owner_limit=maximum_resident_bytes//3,
            reason='one native Feature window exceeds resident budget')
        _guard_resident(combined, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, reason='one native Feature window exceeds resident budget')
        stats['native_window_reads'] += 1
        tick = time.perf_counter_ns()
        native = qlib_inputs['view'].read(fields=FIELDS, symbols=symbols, start=history[0], end=history[-1])
        window = _NativeWindow(native, sessions=history, symbols=symbols,
                               instrument_map=qlib_inputs['view_reference']['instrument_map'])
        del native
        stats['native_window_read_ns'] += time.perf_counter_ns()-tick
        _guard_resident(window.bytes, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, owner_limit=maximum_resident_bytes//3,
            reason='native Feature window exceeds resident budget')
        stats['native_window_peak_rows'] = max(stats['native_window_peak_rows'], len(window.frame))
        stats['native_window_peak_sessions'] = max(stats['native_window_peak_sessions'], len(history))
        stats['native_window_peak_bytes'] = max(stats['native_window_peak_bytes'], window.bytes)
        previous = None
        signature = None
        previous_bytes = sys.getsizeof(None)
        pending = []
        for day in block:
            position = positions[day]
            history = sessions[position-history_sessions+1:position+1]
            cutoffs = {session: config['cutoff_by_session'][day] for session in history}
            price_query = replace(qlib_inputs['price_query'], sessions=history, cutoff_by_session=cutoffs)
            factor_query = replace(qlib_inputs['factor_query'], sessions=history, cutoff_by_session=cutoffs)
            member_query = replace(qlib_inputs['member_query'], sessions=history, cutoff_by_session=cutoffs)
            resident_base = base+window.bytes+previous_bytes
            before_read = resident_base+_pending_core_bytes(pending)+read_estimate+core_estimate+reuse_budget_bytes
            if pending and before_read+_caller_bytes(caller_retained_bytes) > maximum_resident_bytes:
                stats['batch_budget_flushes'] += 1
                completed = yield from _iter_core_feature_batch(pending, resident_base=resident_base,
                    maximum_resident_bytes=maximum_resident_bytes, reuse_budget_bytes=reuse_budget_bytes,
                    stats=stats, caller_retained_bytes=caller_retained_bytes, progress=progress,
                    completed=completed, total=len(outputs), begin=begin)
            retained_pending = base+previous_bytes+_pending_owned_bytes(pending)+reuse_budget_bytes
            _guard_resident(resident_base+_pending_core_bytes(pending)+read_estimate+reuse_budget_bytes,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='one Reader Feature view exceeds resident budget')
            stats['data_read_calls'] += 1
            tick = time.perf_counter_ns()
            source = data.read(snapshot=config['snapshot'], query=price_query)
            stats['data_read_ns'] += time.perf_counter_ns()-tick
            prices, prices_wire = window.project(source,
                view_ref=qlib_inputs['view_reference']['view_id'], stats=stats,
                maximum_resident_bytes=maximum_resident_bytes, retained_bytes=retained_pending,
                caller_retained_bytes=caller_retained_bytes, _with_wire=True)
            del source
            prices_bytes = _batch_bytes(prices,stats=stats)
            _guard_resident(resident_base+_pending_owned_bytes(pending)+reuse_budget_bytes+2*prices_bytes+read_estimate,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='one Reader Feature view exceeds resident budget')
            stats['data_read_calls'] += 1
            tick = time.perf_counter_ns()
            source = data.read(snapshot=config['snapshot'], query=factor_query)
            stats['data_read_ns'] += time.perf_counter_ns()-tick
            # Reader batches are isolated; public adjustment reads its inputs.
            # The local amount-provenance alias is never written in this phase.
            stats['reader_batch_size_reuses'] = stats.get('reader_batch_size_reuses',0)+1
            factors = window.project(source,
                view_ref=qlib_inputs['view_reference']['view_id'], stats=stats,
                maximum_resident_bytes=maximum_resident_bytes,
                retained_bytes=retained_pending+2*prices_bytes+_header_bytes(prices.field_meta),
                caller_retained_bytes=caller_retained_bytes)
            del source
            # Public adjustment serializes source/derivation lineage. Reserve
            # those copies before calling it, rather than after large wires exist.
            factors_bytes = _batch_bytes(factors,stats=stats)
            adjustment_reserve = resident_base+_pending_owned_bytes(pending)+reuse_budget_bytes+24*(prices_bytes+factors_bytes)
            _guard_resident(adjustment_reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature adjustment/source working set exceeds resident budget')
            tick = time.perf_counter_ns()
            adjusted_wire = _adjust_feature_wire(prices, factors, day, prices_wire)
            del prices_wire
            adjusted_bytes = 4*(prices_bytes+factors_bytes)
            reader_bytes = prices_bytes+factors_bytes+adjusted_bytes
            stats['reader_batch_size_reuses'] += 2
            # to_json owns this wire. The adapter only reads it and retains
            # provenance aliases; no Data operation receives this private graph.
            adjusted_wire_bytes = 2*adjusted_bytes
            _guard_resident(resident_base+_pending_owned_bytes(pending)+reuse_budget_bytes+
                reader_bytes+adjusted_wire_bytes+read_estimate,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature membership/source working set exceeds resident budget')
            stats['data_read_calls'] += 1
            membership_tick = time.perf_counter_ns()
            membership = data.members(snapshot=config['snapshot'], query=member_query)
            stats['data_read_ns'] += time.perf_counter_ns()-membership_tick
            stats['reader_batch_size_reuses'] += 3
            membership_bytes = _batch_bytes(membership,stats=stats)
            retained = resident_base+_pending_owned_bytes(pending)+reuse_budget_bytes+reader_bytes
            reserve = retained+adjusted_wire_bytes+2*membership_bytes
            stats['private_wire_size_reuses'] = stats.get('private_wire_size_reuses',0)+1
            _guard_resident(reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature membership/source working set exceeds resident budget')
            membership_wire = membership.to_json()
            wire_estimate = adjusted_wire_bytes+2*membership_bytes
            _guard_resident(retained+3*wire_estimate,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature adapter/wire copies exceed resident budget')
            adapted = _adapt_decision_wires(adjusted_wire, membership_wire,
                recipe_ref=catalog.recipe_ref(config['feature_selection'], normalized=True),
                output_keys=tuple((security, day) for security in symbols), source_granularity='batch_field')
            plan = build_feature_plan(adapted.plan, config['feature_selection'], catalog=catalog, normalized=True)
            documents = [plan, adapted.facts, adapted.context]
            # Serialized lengths plus an eightfold decoded/header allowance;
            # Core cell/dependency workspace is reserved separately below.
            document_estimate = sum(sys.getsizeof(document.payload) for document in documents)*8
            _guard_resident(retained+wire_estimate+document_estimate,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature Core snapshot copies exceed resident budget')
            wires = plan.to_dict(), adapted.facts.to_dict(), adapted.context.to_dict()
            if not config.get("_compact_source",False):
                signature = _view_signature(plan, adapted.facts, adapted.context, calendar=sessions, _wires=wires)
                if previous is not None:
                    classification = classify_shared_feature_views(previous, signature)
                    counts = stats['classifier_status_counts']
                    status = classification['status']
                    counts[status] = counts.get(status, 0)+1
                    counts = stats['classifier_reason_counts']
                    for reason in classification['reasons']:
                        counts[reason] = counts.get(reason, 0)+1
                previous = signature
                previous_bytes = len(signature['facts'])*1024+len(signature['reference'])*512+len(sessions)*256
                stats['signature_charge_estimates'] = stats.get('signature_charge_estimates', 0)+1
            owned = (base+window.bytes+reuse_budget_bytes+_pending_owned_bytes(pending)+reader_bytes+
                     membership_bytes+previous_bytes+wire_estimate+3*document_estimate)
            _guard_resident(owned, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature input working graph exceeds resident budget')
            members = {row['security_id']: row['is_member'] for row in membership_wire['records'] if row['session'] == day}
            if config.get('_compact_source',False):
                stats['feature_source_batch_ref_reuses']=stats.get('feature_source_batch_ref_reuses',0)+2
            evidence = {'session': day, 'core_plan': wires[0],
                'fact_ref': adapted.facts.identity, 'context_ref': adapted.context.identity,
                'sessions': list(history), 'cutoffs': cutoffs,
                'adjusted_input_ref': (next(source['batch_ref'] for source in adapted.source_evidence.values()
                    if source['field']!='is_member') if config.get('_compact_source',False) else digest(adjusted_wire)),
                'membership_ref': (next(source['batch_ref'] for source in adapted.source_evidence.values()
                    if source['field']=='is_member') if config.get('_compact_source',False) else digest(membership_wire)), 'source_evidence': {key: {
                    **{name: value for name, value in source.items() if name != 'provenance_by_key'},
                    'provenance_by_key_ref': digest(source['provenance_by_key'])}
                    for key, source in adapted.source_evidence.items()}}
            workspace, output_reserve = _core_view_bounds(*wires, snapshots=document_estimate)
            current_view = _seal_pending_view((plan, adapted.facts, adapted.context),
                evidence, members, workspace, output_reserve, stats=stats,
                owned_estimate=document_estimate+output_reserve+len(members)*512)
            # Only immutable original Documents and compact saved provenance
            # survive collection. No source DataFrames or full adapter graph do.
            del prices, factors, adjusted_wire, membership, membership_wire, adapted, plan
            del wires, documents, evidence, members
            stats['feature_adaptation_ns'] += time.perf_counter_ns()-tick
            read_estimate = max(read_estimate, owned-(resident_base+_pending_owned_bytes(pending)+reuse_budget_bytes))
            resident_base = base+window.bytes+previous_bytes
            candidate = pending+[current_view]
            must_flush = pending and resident_base+_pending_core_bytes(candidate)+reuse_budget_bytes+_caller_bytes(caller_retained_bytes) > maximum_resident_bytes
            del candidate
            if must_flush:
                stats['batch_budget_flushes'] += 1
                # The just-prepared compact view remains live while older views
                # execute/yield; charge it too. Recheck the caller after flushing.
                completed = yield from _iter_core_feature_batch(pending,
                    resident_base=resident_base+current_view[5],
                    maximum_resident_bytes=maximum_resident_bytes, reuse_budget_bytes=reuse_budget_bytes,
                    stats=stats, caller_retained_bytes=caller_retained_bytes, progress=progress,
                    completed=completed, total=len(outputs), begin=begin)
            pending.append(current_view)
            del current_view
            stats['pending_view_peak_bytes'] = max(stats['pending_view_peak_bytes'], _pending_owned_bytes(pending))
            _guard_resident(base+window.bytes+previous_bytes+_pending_core_bytes(pending)+reuse_budget_bytes,
                maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Core batch snapshots/output working set exceeds resident budget')
        completed = yield from _iter_core_feature_batch(pending,
            resident_base=base+window.bytes+previous_bytes, maximum_resident_bytes=maximum_resident_bytes,
            reuse_budget_bytes=reuse_budget_bytes, stats=stats, caller_retained_bytes=caller_retained_bytes,
            progress=progress, completed=completed, total=len(outputs), begin=begin)
        del window, previous, signature, pending
        start += count
    stats['feature_pipeline_total_ns'] = time.perf_counter_ns()-begin_ns
    stats['feature_pipeline_clock_basis'] = 'elapsed_generator_lifetime_including_caller_pauses'

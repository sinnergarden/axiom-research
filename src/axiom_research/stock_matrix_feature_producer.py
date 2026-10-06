"""Bounded opt-in Feature input preparation through the existing Data/Core path.

Qlib supplies an immutable native projection, not revision selection. Every
output still reads its original history at its declared cutoff and uses Core
for the complete cross-section. Shared-panel classification is diagnostic
until an explicit group execution/evidence contract is admitted.
"""
from __future__ import annotations

from dataclasses import replace
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
    if id(value) in seen:
        return 0
    seen.add(id(value))
    amount = sys.getsizeof(value)
    if isinstance(value, dict):
        amount += sum(_owned_bytes(k, seen)+_owned_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        amount += sum(_owned_bytes(v, seen) for v in value)
    return amount


def _batch_bytes(batch):
    return int(batch.frame.memory_usage(deep=True).sum())+_owned_bytes([batch.field_meta, batch.context])


def _caller_bytes(getter):
    if getter is None:
        return 0
    _require(callable(getter), 'caller_retained_bytes must be callable')
    value = getter()
    _require(type(value) is int and value >= 0, 'caller_retained_bytes must return a nonnegative integer')
    return value


def _guard_resident(owned, *, maximum_resident_bytes, stats, caller_retained_bytes, reason, owner_limit=None):
    """One combined owned-graph guard; neither counter is a RSS measurement."""
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
                caller_retained_bytes=None):
        """Round actual Reader values exactly as the native float32 format.

        Equal native values can be reused. Revision/missingness differences
        use the current Reader value, with its unaltered field provenance.
        This operation performs no Feature or adjustment mathematics.
        """
        import numpy as np
        import pandas as pd
        from axiom_data import DataBatch
        # Reject a large coverage/source graph before to_json duplicates it.
        reserve = self.bytes+retained_bytes+2*_batch_bytes(batch)+len(batch.frame)*len(FIELDS)*32
        _guard_resident(reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, reason='Reader projection working set exceeds resident budget')
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
        context = {**wire['context'], 'numeric_projection': {
            'contract_version': 'research_qlib_native_projection_v2', 'view_ref': view_ref,
            'reader_batch_ref': digest(wire),
            'revision_admission': 'actual_reader_float32_with_native_value_reuse',
            'dtype': 'float32_values_promoted_to_float64'}}
        return DataBatch(frame, batch.field_meta, context)


def _view_signature(plan, facts, context, *, calendar=None):
    """Compact exact input compatibility evidence, without a second executor."""
    p, f, c = plan.to_dict(), facts.to_dict(), context.to_dict()
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


def iter_matrix_feature_days(data, *, config, catalog, chosen, qlib_inputs,
                             history_sessions=21, output_block_sessions=64,
                             maximum_resident_bytes, progress=None, stats=None, caller_retained_bytes=None):
    """Yield original complete daily rows/evidence, with bounded native reads.

    Actual multi-output execution is deliberately zero until its evidence
    bridge is frozen. Each current execution is a genuine existing Core call.
    """
    from .data_adapter import _adapt_decision_wires
    from .feature_catalog import build_feature_plan
    from .stock_ml import _adjust_feature
    from axiom_engine.core import execute_feature_plan
    for value, name in [(history_sessions, 'history_sessions'), (output_block_sessions, 'output_block_sessions'),
                        (maximum_resident_bytes, 'maximum_resident_bytes')]:
        _positive(value, name)
    _require(history_sessions == max(entry['lookback'] for entry in chosen),
             'Feature history differs from selected catalog lookback')
    stats = {} if stats is None else stats
    for name in ('data_read_calls', 'core_calls', 'feature_core_calls', 'native_window_reads',
                 'native_window_peak_rows', 'native_window_peak_sessions', 'native_window_peak_bytes',
                 'working_graph_peak_bytes', 'combined_working_graph_peak_bytes', 'yield_live_bytes',
                 'native_value_reuses', 'reader_projection_fallback_cells',
                 'reader_projection_fallback_null_cells', 'core_single_output_groups',
                 'actual_core_multi_output_groups'):
        stats.setdefault(name, 0)
    stats.setdefault('classifier_status_counts', {})
    stats.setdefault('classifier_reason_counts', {})
    stats['group_execution_status'] = 'ORIGINAL_DAILY_CORE_PENDING_GROUP_EVIDENCE'
    _guard_resident(0, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                    caller_retained_bytes=caller_retained_bytes, reason='caller working set exceeds resident budget')
    symbols = tuple(config['symbols'])
    sessions = tuple(config['read_sessions'])
    outputs = list(config['feature_sessions'])
    positions = {day: index for index, day in enumerate(sessions)}
    _require(outputs == sorted(set(outputs)) and set(outputs) <= set(sessions),
             'ordered covered matrix Feature outputs required')
    _require(all(positions[day]+1 >= history_sessions for day in outputs), 'Feature lookback incomplete')
    begin = time.perf_counter()
    start = 0
    completed = 0
    while start < len(outputs):
        count = min(output_block_sessions, len(outputs)-start)
        while True:
            block = outputs[start:start+count]
            history = sessions[positions[block[0]]-history_sessions+1:positions[block[-1]]+1]
            # Reserve room for Reader, derived wires and Core snapshots too.
            estimate = len(history)*len(symbols)*(512+32*len(FIELDS))
            if estimate <= maximum_resident_bytes//3 or count == 1:
                break
            count = max(1, count//2)
        _guard_resident(estimate, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, owner_limit=maximum_resident_bytes//3,
            reason='one native Feature window exceeds resident budget')
        stats['native_window_reads'] += 1
        native = qlib_inputs['view'].read(fields=FIELDS, symbols=symbols, start=history[0], end=history[-1])
        window = _NativeWindow(native, sessions=history, symbols=symbols,
                               instrument_map=qlib_inputs['view_reference']['instrument_map'])
        del native
        _guard_resident(window.bytes, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
            caller_retained_bytes=caller_retained_bytes, owner_limit=maximum_resident_bytes//3,
            reason='native Feature window exceeds resident budget')
        stats['native_window_peak_rows'] = max(stats['native_window_peak_rows'], len(window.frame))
        stats['native_window_peak_sessions'] = max(stats['native_window_peak_sessions'], len(history))
        stats['native_window_peak_bytes'] = max(stats['native_window_peak_bytes'], window.bytes)
        previous = None
        for day in block:
            position = positions[day]
            history = sessions[position-history_sessions+1:position+1]
            cutoffs = {session: config['cutoff_by_session'][day] for session in history}
            price_query = replace(qlib_inputs['price_query'], sessions=history, cutoff_by_session=cutoffs)
            factor_query = replace(qlib_inputs['factor_query'], sessions=history, cutoff_by_session=cutoffs)
            member_query = replace(qlib_inputs['member_query'], sessions=history, cutoff_by_session=cutoffs)
            previous_bytes = _owned_bytes(previous)
            stats['data_read_calls'] += 1
            prices = window.project(data.read(snapshot=config['snapshot'], query=price_query),
                view_ref=qlib_inputs['view_reference']['view_id'], stats=stats,
                maximum_resident_bytes=maximum_resident_bytes, retained_bytes=previous_bytes,
                caller_retained_bytes=caller_retained_bytes)
            stats['data_read_calls'] += 1
            factors = window.project(data.read(snapshot=config['snapshot'], query=factor_query),
                view_ref=qlib_inputs['view_reference']['view_id'], stats=stats,
                maximum_resident_bytes=maximum_resident_bytes, retained_bytes=previous_bytes+_batch_bytes(prices),
                caller_retained_bytes=caller_retained_bytes)
            adjusted, adjusted_wire = _adjust_feature(prices, factors, day, _with_wire=True)
            stats['data_read_calls'] += 1
            membership = data.members(snapshot=config['snapshot'], query=member_query)
            retained = window.bytes+previous_bytes+sum(_batch_bytes(batch) for batch in (prices, factors, adjusted))
            reserve = retained+_owned_bytes(adjusted_wire)+2*_batch_bytes(membership)
            _guard_resident(reserve, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature membership/source working set exceeds resident budget')
            membership_wire = membership.to_json()
            adapted = _adapt_decision_wires(adjusted_wire, membership_wire,
                recipe_ref=catalog.recipe_ref(config['feature_selection'], normalized=True),
                output_keys=tuple((security, day) for security in symbols), source_granularity='batch_field')
            plan = build_feature_plan(adapted.plan, config['feature_selection'], catalog=catalog, normalized=True)
            signature = _view_signature(plan, adapted.facts, adapted.context, calendar=sessions)
            if previous is not None:
                classification = classify_shared_feature_views(previous, signature)
                counts = stats['classifier_status_counts']
                status = classification['status']
                counts[status] = counts.get(status, 0)+1
                counts = stats['classifier_reason_counts']
                for reason in classification['reasons']:
                    counts[reason] = counts.get(reason, 0)+1
            previous = signature
            owned = window.bytes+sum(int(batch.frame.memory_usage(deep=True).sum())
                for batch in (prices, factors, adjusted, membership))+_owned_bytes([
                    *[[batch.field_meta, batch.context] for batch in (prices, factors, adjusted, membership)],
                    adjusted_wire, membership_wire, signature, plan.payload, adapted.facts.payload,
                    adapted.context.payload, adapted.plan.payload, plan.to_dict(), adapted.facts.to_dict(),
                    adapted.context.to_dict(), adapted.source_evidence])
            _guard_resident(owned, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature input working graph exceeds resident budget')
            frame = execute_feature_plan(plan, adapted.facts, adapted.context)
            stats['core_calls'] += 1
            stats['feature_core_calls'] += 1
            stats['core_single_output_groups'] += 1
            frame_wire = frame.to_dict()
            members = {row['security_id']: row['is_member'] for row in membership_wire['records'] if row['session'] == day}
            rows = [{'security_id': row['security_id'], 'session': day, 'values': row['values'],
                'availability': row['availability'], 'validity': row['valid'], 'reasons': row['reasons'],
                'member': members[row['security_id']], 'knowledge_cutoff': config['cutoff_by_session'][day],
                'source_refs': [frame.identity, plan.identity]} for row in frame_wire['rows']]
            evidence = {'session': day, 'core_frame_ref': frame.identity, 'core_plan': plan.to_dict(),
                'fact_ref': adapted.facts.identity, 'context_ref': adapted.context.identity,
                'sessions': list(history), 'cutoffs': cutoffs, 'adjusted_input_ref': digest(adjusted_wire),
                'membership_ref': digest(membership_wire), 'source_evidence': {key: {
                    **{name: value for name, value in source.items() if name != 'provenance_by_key'},
                    'provenance_by_key_ref': digest(source['provenance_by_key'])}
                    for key, source in adapted.source_evidence.items()}}
            owned += _owned_bytes([frame.payload, frame_wire, rows, evidence])
            _guard_resident(owned, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature output working graph exceeds resident budget')
            # Release all temporary source/panel/Core carriers before yielding.
            del prices, factors, adjusted, adjusted_wire, membership, membership_wire, adapted, plan, frame, frame_wire
            completed += 1
            if progress is not None:
                progress({'stage': 'matrix_features', 'completed': completed, 'total': len(outputs),
                          'session': day, 'seconds': time.perf_counter()-begin})
            live = window.bytes+_owned_bytes([signature, rows, evidence])
            _guard_resident(live, maximum_resident_bytes=maximum_resident_bytes, stats=stats,
                caller_retained_bytes=caller_retained_bytes, reason='Feature yield working graph exceeds resident budget')
            stats['yield_live_bytes'] = live
            try:
                yield rows, evidence
            finally:
                stats['yield_live_bytes'] = 0
            del rows, evidence
        del window, previous
        start += count

"""Shared Data columns to the existing Core DAG and saved Feature owner."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time

from .stock_artifacts import digest, file_digest, _read, _verify_ref
from .stock_fold_inputs import require, seal

ARRAY_KEYS = {'values', 'validity', 'reason_codes', 'available_at_utc_us', 'source_codes'}


def column_result_rows(result, selection, member):
    """Convert only the requested small output, preserving the owner header."""
    day = selection['session']
    require(result['sessions'] == [day], 'one daily anchored column output required')
    header = {k: v for k, v in result.items() if k not in ARRAY_KEYS | {'stats'}}
    reasons = result['reason_dictionary']; null = result['null_available_at_utc_us']
    width = len(result['schema']); dates = {}
    def instant(value):
        value = int(value)
        if value == null: return None
        if value not in dates:
            dates[value] = datetime.fromtimestamp(value / 1000000, timezone.utc).isoformat().replace('+00:00', 'Z')
        return dates[value]
    rows = []
    for i, security in enumerate(result['security_ids']):
        valid = [bool(v) for v in result['validity'][0, i]]
        rows.append({'security_id': security, 'session': day,
            'values': [float(result['values'][0, i, j]) if valid[j] else None for j in range(width)],
            'validity': valid, 'availability': [instant(v) for v in result['available_at_utc_us'][0, i]],
            'reasons': [[] if int(v) == 0 else [reasons[int(v)]] for v in result['reason_codes'][0, i]],
            'member': bool(member[i]), 'knowledge_cutoff': selection['knowledge_cutoff'],
            'source_refs': [result['result_ref'], result['plan_ref']]})
    return rows, {'selection': selection, 'selection_ref': digest(selection), 'core_header': header}


def iter_column_feature_days(source, *, catalog, sessions, maximum_resident_bytes, stats):
    from axiom_engine.core import compile_feature_columns, execute_feature_columns
    from .stock_feature_columns import feature_column_plan
    plan = feature_column_plan(source, catalog); wire = plan.to_dict()
    tick = time.perf_counter_ns(); compiled = compile_feature_columns(plan)
    stats['core_compile_calls'] = 1; stats['core_compile_ns'] = time.perf_counter_ns() - tick
    for day in sessions:
        tick = time.perf_counter_ns()
        inputs, selection = source.core_inputs(day, calendar_ref=wire['calendar_ref'],
            reference_ref=wire['reference_ref'], schema=wire['input_schema'])
        stats['column_view_ns'] = stats.get('column_view_ns', 0) + time.perf_counter_ns() - tick
        tick = time.perf_counter_ns()
        result = execute_feature_columns(compiled, inputs=inputs, output_sessions=[day],
            max_working_bytes=maximum_resident_bytes)
        stats['feature_core_calls'] += 1
        stats['core_public_ns'] = stats.get('core_public_ns', 0) + time.perf_counter_ns() - tick
        for key in ('admission_ns', 'numeric_and_metadata_ns', 'output_ns'):
            stats['core_' + key] = stats.get('core_' + key, 0) + result['stats'][key]
        stats['core_peak_working_bytes'] = max(stats.get('core_peak_working_bytes', 0), result['stats']['peak_working_bytes'])
        rows, proof = column_result_rows(result, selection, inputs['reference_member'][-1])
        del result, inputs
        yield rows, proof


def build_column_feature_matrix(data, *, spec, destination, storage_options, progress, column_source):
    from .feature_catalog import load_feature_catalog_ref
    from .stock_feature_columns import build_feature_column_source, load_feature_column_source
    from .stock_feature_inputs import (_spec, _storage_options, _qlib, _qlib_scope,
        load_stock_feature_inputs, _atomic, _publish_compact_index, DEFAULT_LIMITS)
    from .stock_ml import _implementation, _environment
    from .stock_matrix_storage import row_index, write_part
    from .stock_matrix_reader import VerifiedMatrixStore, admit_feature_matrix_parts, _published_feature_matrix
    from .stock_matrix_feature_column_metadata import write_block

    spec = deepcopy(spec); options = _storage_options(storage_options)
    universe_id = _spec(spec); catalog = load_feature_catalog_ref(spec['catalog_ref'])
    chosen = catalog.select(spec['feature_selection'])
    require([c['id'] for c in chosen] == spec['ordered_features'], 'column Feature catalog order mismatch')
    source_path = Path(column_source) if column_source is not None else Path(destination) / 'shared-source/source.json'
    if source_path.exists():
        source = load_feature_column_source(source_path); source_metrics = {'data_read_calls': 0, 'reused': True}
    else:
        require(column_source is None, 'explicit saved column source does not exist')
        source, source_metrics = build_feature_column_source(data, spec=spec, destination=source_path.parent, progress=progress)
    physical_spec=source.manifest['spec']
    require(all(physical_spec[k]==spec[k] for k in ('calendar','universe','snapshot','pit_policy','scope')) and
        set(spec['read_sessions'])<=set(physical_spec['read_sessions']) and
        set(spec['feature_sessions'])<=set(physical_spec['feature_sessions']) and
        all(physical_spec['cutoff_by_session'][d]==spec['cutoff_by_session'][d] for d in spec['read_sessions']),
        'shared column source axes/config mismatch')
    source_desc = {'path': str(source_path.resolve()), 'file_digest': file_digest(source_path), 'source_ref': source.identity}
    impl = _implementation()
    definition = {'spec': spec, 'storage_options': options, 'implementation_sources': impl,
        'implementation_ref': digest(impl), 'environment': _environment(), 'column_source': source_desc}
    ref = digest(definition); target = Path(destination) / ref[7:]
    if (target / 'index.json').exists():
        saved = load_stock_feature_inputs(target)
        require(saved.to_dict()['definition'] == definition, 'saved column Feature definition mismatch')
        saved.reused = True
        return saved
    target.mkdir(parents=True, exist_ok=True)
    view = source.manifest['qlib_view']; _qlib_scope(view, physical_spec, universe_id)
    manifests = list((source_path.parent / 'qlib').glob('*/axiom-qlib.json'))
    require(len(manifests) == 1, 'one fixed shared Qlib manifest required')
    qdesc = {'path': str(manifests[0].resolve()), 'file_digest': file_digest(manifests[0]), 'view_id': view['view_id']}
    schema = [{'name': c['id'], 'dtype': 'float64', 'unit': 'dimensionless',
        'stage': 'cross_sectional', 'missing': 'preserve'} for c in chosen]
    row_wire = row_index(spec['feature_sessions'], spec['universe']); rid = write_part(target, row_wire, 'row_index_ref')
    store = VerifiedMatrixStore(maximum_matrix_bytes=options['maximum_resident_bytes'],
        maximum_parent_bytes=DEFAULT_LIMITS['maximum_parent_bytes'])
    store._begin_publication_accounting()
    try:
        _spec(spec,reader=store.read_json)
        require(_qlib(qdesc, DEFAULT_LIMITS, fingerprints=store.fingerprints,
            _file_hasher=store._hash_once, _json_reader=store._decode_once) == view, 'shared Qlib manifest mismatch')
        store.read_json(rid, 'row_index_ref')
        partitions = []; blocks = []; selections = []; reachable = set(); parents = {}; completed = []
        checkpoint = target / 'checkpoint.json'
        if checkpoint.exists():
            old = _read(checkpoint); _verify_ref(old, 'content_digest')
            require(old['contract_version'] == 'stock_feature_inputs_checkpoint_v4' and
                old['definition_ref'] == ref and old['qlib_manifest'] == qdesc and
                old['schema'] == schema and old['row_index'] == rid, 'column Feature checkpoint mismatch')
            partitions = old['partitions']
            completed, blocks, reachable, parents = admit_feature_matrix_parts(store, spec=spec, view=view,
                schema=schema, row_index=row_wire, partitions=partitions, complete=False,
                universe_id=universe_id, metadata_version='stock_matrix_feature_metadata_v3',column_source=source_desc)
            selections = [r for b in blocks for r in b['_source_selection_rows']]
            require(selections == old['feature_rows'], 'column Feature checkpoint selection mismatch')
        stats = {'source_preparation': source_metrics, 'data_read_calls': 0, 'feature_core_calls': 0,
            'feature_writer_blocks': 0, 'feature_writer_ns': 0}
        rows = []; proof = []
        def publish():
            nonlocal rows, proof
            tick = time.perf_counter_ns()
            new_parts, new_selection = write_block(target, rows=rows, proof=proof,
                source_descriptor=source_desc, spec=spec, view=view, schema=schema,
                index_ref=row_wire['row_index_ref'], row_offset=len(completed) * len(spec['universe']),
                options=options, universe_id=universe_id, source_store=store)
            days, new_blocks, hashes, new_parents = admit_feature_matrix_parts(store, spec=spec, view=view,
                schema=schema, row_index=row_wire, partitions=new_parts, complete=False,
                universe_id=universe_id, start_offset=len(completed) * len(spec['universe']),
                metadata_version='stock_matrix_feature_metadata_v3',column_source=source_desc)
            require(new_selection == [r for b in new_blocks for r in b['_source_selection_rows']], 'column output admission mismatch')
            partitions.extend(new_parts); blocks.extend(new_blocks); selections.extend(new_selection)
            completed.extend(days); reachable.update(hashes); parents.update(new_parents)
            _atomic(checkpoint, seal({'contract_version': 'stock_feature_inputs_checkpoint_v4',
                'definition_ref': ref, 'qlib_manifest': qdesc, 'schema': schema, 'row_index': rid,
                'partitions': partitions, 'feature_rows': selections}, 'content_digest'))
            rows = []; proof = []
            stats['feature_writer_blocks'] += 1; stats['feature_writer_ns'] += time.perf_counter_ns() - tick
            if progress is not None:
                progress({'stage': 'feature_column_block', 'completed_dates': len(completed),
                    'total_dates': len(spec['feature_sessions']), 'producer_stats': deepcopy(stats)})
        for daily, evidence in iter_column_feature_days(source, catalog=catalog,
                sessions=spec['feature_sessions'][len(completed):], maximum_resident_bytes=options['maximum_resident_bytes'], stats=stats):
            rows.extend(daily); proof.append(evidence)
            if len(proof) == options['row_block_sessions']: publish()
        if proof: publish()
        require(completed == spec['feature_sessions'], 'incomplete column Feature output')
        selection = write_part(target, seal({'contract_version': 'stock_feature_source_selection_v3',
            'definition_ref': ref, 'feature_rows': selections}, 'source_selection_ref'), 'source_selection_ref')
        value = {'contract_version': 'stock_feature_inputs_v4', 'definition': definition, 'definition_ref': ref,
            'status': 'COMPLETE', 'qlib_view': view, 'qlib_manifest': qdesc, 'schema': schema,
            'schema_digest': digest(schema), 'row_index': rid, 'source_selection': selection, 'partitions': partitions}
        value['feature_inputs_ref'] = digest({k: value[k] for k in ('contract_version', 'definition_ref',
            'qlib_manifest', 'schema_digest', 'row_index', 'source_selection', 'partitions')})
        _publish_compact_index(target / 'index.json', seal(value, 'content_digest'))
        return _published_feature_matrix(target, value, store, row_wire, blocks, reachable, parents)
    except BaseException:
        store.close()
        raise

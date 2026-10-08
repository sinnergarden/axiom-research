"""One Research Raw operator and existing Core CS; compact v3 publication."""
from array import array
from copy import deepcopy
from dataclasses import replace, fields as dataclass_fields
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
from pathlib import Path
import errno
import json
import os
import sys
import tempfile
import time
from weakref import WeakKeyDictionary
from contextlib import contextmanager

from .stock_artifacts import digest, digest_array_rows, file_digest, write_json
from .stock_fold_inputs import require, seal, validate_spec
from .stock_label_contracts import (_eligible_reason, _instant, NORMALIZATION_SPEC,
    RAW_TARGET_SCHEMA, NORMALIZED_TARGET_SCHEMA, _normalization_sources, core_clock)
from .stock_matrix_storage import write_buffer, write_part, instant_us
from .stock_compact_store import (_view_data, StockFeatureView, load_stock_feature_view,
    iter_feature_rows, iter_feature_eligibility, set_feature_window, OwnedStore, limits, sealed, _size)

_RAW_DOMAINS = WeakKeyDictionary()
_RAW_DOMAIN_TOKEN = object()


class _RawPriceDomain:
    """Private ownership of detached Data JSON and one lazy Raw key index."""
    __slots__ = ('__weakref__',)

    def __init__(self, token, value):
        require(token is _RAW_DOMAIN_TOKEN, 'private Raw price domain required')
        _RAW_DOMAINS[self] = value

    def _data(self):
        require(self in _RAW_DOMAINS, 'Raw price domain is closed')
        value = _RAW_DOMAINS[self]
        require(value['pid'] == os.getpid(), 'Raw price domain belongs to another process')
        return value

    @property
    def source(self):
        return deepcopy(self._data()['source'])

    @property
    def source_ref(self):
        return self._data()['source']['price_view_ref']

    def require_request(self, spec, cutoff, days):
        value = self._data()
        require(value['binding'] == (spec['snapshot'], spec['pit_policy'], tuple(spec['calendar']),
                                     tuple(spec['universe'])) and
                _instant(cutoff) == value['parsed'][3] and set(days) <= value['features'],
                'Raw price domain request/cutoff mismatch')

    @staticmethod
    def _roots(value):
        # Explicit graphs, including indexed aliases; never charge an opaque
        # Python handle as if it contained no rows. Metrics are external roots.
        return [value[k] for k in ('source', 'binding', 'features', 'parsed',
                                   'records', 'field_meta', 'compiled')]

    def _charge(self):
        value = self._data(); metrics = value['metrics']
        old = metrics['_price_view_bytes']; metrics['_price_view_bytes'] = 0
        try:
            charge = _measured(metrics, self._roots(value)) + sys.getsizeof(value) + 512
            _working(metrics, charge)
        except BaseException:
            metrics['_price_view_bytes'] = old
            _sync_feature_charge(metrics)
            raise
        metrics['_price_view_bytes'] = charge
        _sync_feature_charge(metrics)
        return charge

    def _charge_compiled(self):
        """Charge only allocations added to the admitted native graph."""
        value = self._data(); metrics = value['metrics']; compiled = value['compiled']
        indexes = (compiled['indexed'], *compiled['metadata'].values())
        # Values and tuple members alias admitted rows/strings. Each _keyed
        # inserts a fresh pair; all such pairs have this fixed Python size.
        # The small container set also prevents charging any container alias
        # twice. No row/provenance graph traversal belongs at this boundary.
        seen = set(); increment = 1024; key_count = 0
        for container in (compiled, compiled['metadata'], *indexes, compiled['positions']):
            if id(container) not in seen:
                seen.add(id(container)); increment += sys.getsizeof(container)
        seen_indexes = set()
        for index in indexes:
            if id(index) not in seen_indexes:
                seen_indexes.add(id(index)); key_count += len(index)
        increment += key_count * sys.getsizeof((None, None))
        # Enumerate integers may alias cached native scalars. Charging all of
        # them at the largest position size is a conservative upper bound;
        # the 1024 bytes above bound the few new control strings/headers.
        increment += len(compiled['positions']) * sys.getsizeof(max(0, len(compiled['positions']) - 1))
        _working(metrics, increment)
        metrics['_price_view_bytes'] += increment
        _sync_feature_charge(metrics)
        metrics['price_domain_index_increment_bytes'] = metrics.get('price_domain_index_increment_bytes', 0) + increment
        return metrics['_price_view_bytes']

    def _release_native_lists(self):
        value = self._data(); metrics = value['metrics']; released = 0; seen = set()
        # Retain the admitted headers/empty containers. Only list capacity is
        # released; all row/provenance values remain owned by the indexes.
        for rows in (value['records'], *((value['field_meta'].get(field) or {}).get('by_key')
                                       for field in ('open', 'close'))):
            if rows is not None and id(rows) not in seen:
                seen.add(id(rows)); before = sys.getsizeof(rows)
                rows.clear(); released += before - sys.getsizeof(rows)
        metrics['_price_view_bytes'] -= released
        _sync_feature_charge(metrics)
        metrics['price_domain_released_list_capacity_bytes'] = metrics.get('price_domain_released_list_capacity_bytes', 0) + released
        metrics['price_domain_released_list_containers'] = metrics.get('price_domain_released_list_containers', 0) + len(seen)

    @staticmethod
    def _clear_compiled(compiled):
        if compiled is not None:
            compiled['indexed'].clear()
            for index in compiled['metadata'].values(): index.clear()
            compiled.clear()

    def _borrow(self, calendar, features, source_ref, horizon_sessions=5):
        value = self._data(); metrics = value['metrics']
        require(tuple(calendar) == value['binding'][2] and set(features) <= value['features'] and
                source_ref == value['source']['price_view_ref'] and horizon_sessions == 5,
                'Raw price domain selector/source mismatch')
        if value['compiled'] is None:
            from .labels import _compile_forward_index
            # Keys/set + three dict indexes, including table resizing, are
            # additional to the already charged native JSON graph.
            count = len(value['parsed'][0]['sessions']) * len(value['parsed'][1])
            reserve = count * 512 + len(calendar) * 192 + 65536
            _working(metrics, reserve)
            metrics['price_domain_index_preflight_bytes'] = max(
                metrics.get('price_domain_index_preflight_bytes', 0), reserve)
            metrics['price_domain_index_build_attempts'] = metrics.get('price_domain_index_build_attempts', 0) + 1
            try:
                value['compiled'] = _compile_forward_index(value['records'], value['field_meta'], None,
                    calendar, parsed_query=value['parsed'])
                peak = self._charge_compiled()
            except BaseException:
                self._clear_compiled(value['compiled'])
                value['compiled'] = None
                raise
            metrics['price_domain_index_builds'] = metrics.get('price_domain_index_builds', 0) + 1
            metrics['price_domain_indexed_rows'] = metrics.get('price_domain_indexed_rows', 0) + sum(
                len(value['compiled'][k]) if k == 'indexed' else sum(len(v) for v in value['compiled'][k].values())
                for k in ('indexed', 'metadata'))
            metrics['price_domain_compile_peak_bytes'] = max(metrics.get('price_domain_compile_peak_bytes', 0), peak)
            self._release_native_lists()
        # Check after lazy compilation, before the generator materializes its
        # detached output. A retained index and its child rows coexist.
        _working(metrics, len(features) * len(value['parsed'][1]) * 2048 + 65536)
        value['borrowers'] += 1
        return value['compiled']

    def _release(self):
        value = self._data()
        require(value['borrowers'] > 0, 'Raw price domain borrower underflow')
        value['borrowers'] -= 1

    def close(self):
        value = self._data()
        require(value['borrowers'] == 0, 'Raw price domain is still borrowed')
        metrics = value['metrics']; charge = metrics['_price_view_bytes']
        # Empty private maps as well as dropping roots, so a failed operator's
        # traceback cannot retain the complete domain through an index alias.
        compiled = value['compiled']
        self._clear_compiled(compiled)
        value.clear(); del _RAW_DOMAINS[self]
        compiled = value = None
        metrics['_price_view_bytes'] = 0
        _sync_feature_charge(metrics)
        metrics['price_domain_release_calls'] = metrics.get('price_domain_release_calls', 0) + 1
        metrics['price_domain_released_bytes'] = metrics.get('price_domain_released_bytes', 0) + charge

    def __enter__(self):
        self._data(); return self

    def __exit__(self, *args):
        self.close()


def _implementation(*,columnar=False):
    import inspect
    sources={name:file_digest(Path(__file__).with_name(name)) for name in (
        'labels.py','stock_label_contracts.py','stock_compact_labels.py',
        'stock_compact_store.py','stock_compact_batch.py','stock_batch.py',
        'stock_fold_inputs.py','stock_matrix_storage.py','stock_compact_controls.py')}
    sources['canonical_training_binding']=inspect.getsource(digest_array_rows)
    if columnar:
        sources.update({name:file_digest(Path(__file__).with_name(name)) for name in
            ('stock_column_inputs.py','stock_target_spec.py','stock_training_blocks.py')})
    return digest(sources)


def _input_environment():
    """Only libraries used to prepare inputs; model versions are downstream."""
    import platform
    import importlib.metadata
    return {'python':platform.python_version(), 'packages':{
        p:importlib.metadata.version(p) for p in ('numpy','pandas','pyarrow')}}


def _raw_implementation():
    """Raw identity excludes Feature/cohort/normalization/storage code."""
    import inspect
    from . import labels
    return digest({'operator_version':'forward_5_session_open_close_v1',
        'price_basis':labels.PRICE_BASIS,'functions':{name:inspect.getsource(getattr(labels,name))
        for name in ('_session','_instant','_sessions','_query_context','_keyed','_endpoint',
                     '_compile_forward_index','_forward_rows')},
        'compiled_domain_owner':inspect.getsource(_RawPriceDomain)})


def _column_raw_implementation():
    """Bind the actual Core array kernel and the new endpoint qualifier only."""
    import inspect
    from axiom_engine.core import execute_forward_returns
    from axiom_engine.core.cs_batch import _snapshot_buffer,_descriptor
    from axiom_engine.core.cs_batch import _BUFFER_TYPES
    from axiom_engine.core import contracts
    from importlib.metadata import version
    from .stock_column_inputs import ColumnPriceDomain,query_binding
    return digest({'operator':'core_forward_returns_batch',
        'kernel':inspect.getsource(execute_forward_returns),
        'buffer_admission':inspect.getsource(_snapshot_buffer),
        'buffer_descriptor':inspect.getsource(_descriptor),
        'buffer_types':{name:_BUFFER_TYPES[name] for name in ('values','value_validity')},
        'core_contracts':file_digest(contracts.__file__),'numpy':version('numpy'),
        'qualification':inspect.getsource(ColumnPriceDomain.rows),
        'dependency_binding':inspect.getsource(ColumnPriceDomain._endpoint_dependency),
        'query_binding':inspect.getsource(query_binding)})


def _column_normalization_implementation():
    """The actual CS kernel and eligibility dependencies, not the Core repo."""
    import inspect
    from axiom_engine.core import cs_batch,contracts,plan,execution
    from .stock_target_spec import eligible_target_reason
    from .stock_column_inputs import ColumnMathReuse
    from importlib.metadata import version
    import platform
    return digest({'operator':'core_cs_zscore_batch_v1','core':{
        name:file_digest(module.__file__) for name,module in
        (('cs_batch',cs_batch),('contracts',contracts),('plan',plan))},
        'arithmetic':{name:inspect.getsource(getattr(execution,name)) for name in
            ('_Cell','_merge','_std','_cs_zscore_scale','_cs_zscore_value')},
        'eligibility':inspect.getsource(eligible_target_reason),
        'normalization_projection':inspect.getsource(_normalized),
        'numerical_binding':inspect.getsource(ColumnMathReuse.vector_ref),
        'numpy':version('numpy'),'python':platform.python_version()})


def _working(metrics, amount):
    """Measure each completed chunk once; never rescan an accumulated panel."""
    _sync_feature_charge(metrics)
    total=metrics['_feature_bytes']+metrics['_store'].resident_bytes+metrics['_retained_raw_bytes']+metrics.get('_price_view_bytes',0)+amount
    require(total<=metrics['_limits']['maximum_matrix_bytes'],'compact producer working byte budget exceeded')
    metrics['maximum_working_bytes']=max(metrics['maximum_working_bytes'],total)


def _measured(metrics,value):
    _sync_feature_charge(metrics)
    try:
        return _size(value,maximum=metrics['_limits']['maximum_matrix_bytes'],
            retained=metrics['_feature_bytes']+metrics['_store'].resident_bytes+metrics['_retained_raw_bytes']+metrics.get('_price_view_bytes',0))
    finally:
        value=None


def _raw_reuse_charge(origin, feature_store):
    """Live borrowed owner bytes; one shared Feature handle is counted once."""
    if origin is None: return 0,0
    require(not origin.closed,'Raw reuse source is closed')
    source_feature=_view_data(origin.feature,check=False)['store']
    resident=origin.store.resident_bytes+origin.store.lease_bytes+origin._fixed_shared_bytes
    source=origin.store.metrics['source_bytes']+origin._fixed_shared_source_bytes
    if source_feature is not feature_store:
        resident+=source_feature.resident_bytes+source_feature.lease_bytes
        source+=source_feature.metrics['source_bytes']
    return resident,source


def _sync_feature_charge(metrics):
    """A sequential Feature window changes the live shared byte charge."""
    feature_store=metrics['_feature_store']
    reused,reused_source=_raw_reuse_charge(metrics.get('_raw_origin'),feature_store)
    state=metrics.get('_owner_state')
    external=0 if state is None else max(0,state._fixed_shared_bytes-metrics['_owner_shared_baseline'])
    metrics['raw_reuse_live_bytes']=reused
    metrics['_feature_bytes']=feature_store.resident_bytes+metrics['_caller_bytes']+reused+metrics.get('_checkpoint_bytes',0)+external+metrics.get('_column_math_bytes',0)
    metrics['_store'].shared_bytes=(metrics['_feature_bytes']+metrics['_retained_raw_bytes']+
                                   metrics.get('_price_view_bytes',0))
    metrics['_store'].shared_source_bytes=feature_store.metrics['source_bytes']+metrics['_caller_source_bytes']+reused_source


def _producer_feature_window(feature, offsets, metrics):
    feature_store=metrics['_feature_store']; store=metrics['_store']
    _sync_feature_charge(metrics)
    external=store.shared_bytes-feature_store.resident_bytes
    external_source=store.shared_source_bytes-feature_store.metrics['source_bytes']
    previous=(feature_store.limits,feature_store.shared_bytes,feature_store.shared_source_bytes)
    try:
        feature_store.limits={k:min(v,metrics['_limits'][k]) for k,v in previous[0].items()}
        # external already includes the live Raw/domain graphs charged above.
        feature_store.shared_bytes=max(previous[1],external+store.resident_bytes)
        feature_store.shared_source_bytes=max(previous[2],external_source+store.metrics['source_bytes'])
        selection=metrics.get('_model_feature_selection')
        set_feature_window(feature,offsets,**({} if selection is None else {'model_feature_selection':selection}))
    finally:
        feature_store.limits,feature_store.shared_bytes,feature_store.shared_source_bytes=previous
    _sync_feature_charge(metrics)


def _clock(value):
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=int(value))).isoformat().replace('+00:00','Z')


def _write(root, definition, rows, *, normalized=False, core_ref=None, cohort=None, final_root=None):
    """One typed target, no Raw rows/proof JSON copies at the next stages."""
    reasons=sorted({r['invalid_reason'] for r in rows if r['invalid_reason'] is not None})
    dictionary=[None,*reasons]; codes={r:i for i,r in enumerate(dictionary)}
    calendar=definition['calendar']; positions={d:i for i,d in enumerate(calendar)}
    columns={'values':('float64_le',[r['return'] if r['valid'] else 0.0 for r in rows]),
        'validity':('bool_u8',[r['valid'] for r in rows]),
        'availability':('int64_le',[instant_us(r['label_available_at']) if r['label_available_at'] else 0 for r in rows]),
        'availability_validity':('bool_u8',[r['label_available_at'] is not None for r in rows]),
        'reason_codes':('int32_le',[codes[r['invalid_reason']] for r in rows])}
    sources=[]
    if not normalized:
        columns['start_session']=('int32_le',[positions.get(r['start_session'],-1) for r in rows])
        columns['end_session']=('int32_le',[positions.get(r['end_session'],-1) for r in rows])
        sources=sorted({tuple(r['source_refs']) for r in rows}); source_codes={r:i for i,r in enumerate(sources)}
        columns['source_codes']=('int32_le',[source_codes[tuple(r['source_refs'])] for r in rows])
    buffers={name:write_buffer(root,values,dtype=dtype,shape=[len(rows)]) for name,(dtype,values) in columns.items()}
    if final_root is not None:
        for descriptor in buffers.values():
            descriptor['path']=str(Path(final_root)/Path(descriptor['path']).relative_to(Path(root).resolve()))
    columnar='label_definition_ref' in definition or definition.get('normalization_proof_version')=='stock_normalization_reuse_v1'
    value=seal({'contract_version':('stock_compact_normalized_v2' if columnar else 'stock_compact_normalized_v1')
        if normalized else ('stock_compact_raw_v2' if columnar else 'stock_compact_raw_v1'),
        'definition':definition,'definition_ref':digest(definition),'row_count':len(rows),
        'reason_dictionary':dictionary,'source_dictionary':[list(r) for r in sources],
        'buffers':buffers,'core_ref':core_ref,'cohort':cohort},'target_ref')
    path=Path(root)/'target.json'; write_json(path,value)
    return {'path':str(path.resolve()),'file_digest':file_digest(path),'target_ref':value['target_ref']}


def _publish(target,definition,rows,*,budgets,normalized=False,core_ref=None,cohort=None,store=None):
    """Only a fully checked target directory is made visible at its key."""
    from .stock_compact_batch import read_target
    target=Path(target).resolve(); target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compact-target-',dir=target.parent) as temporary:
        stage=Path(temporary)/'complete'; stage.mkdir()
        desc=_write(stage,definition,rows,normalized=normalized,core_ref=core_ref,cohort=cohort,final_root=target)
        final={'path':str(target/'target.json'),'file_digest':file_digest(stage/'target.json'),'target_ref':desc['target_ref']}
        checker=store or OwnedStore(budgets); original=checker.resolve
        checker.resolve=lambda p:stage/Path(p).relative_to(target) if Path(p).is_relative_to(target) else original(p)
        try:
            read_target(checker,final,expected=definition)
            try: stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
                # An independently published inode requires independent byte
                # admission. Do not pin our unpublished stage to that winner.
                raise ValueError('concurrent compact target publication; fresh admission required') from exc
            checker.resolve=original; checker.check()
        finally:
            checker.resolve=original
            if store is None: checker.close()
        return final


def _source_view(records,meta,context):
    # Bind the actually selected input once. Never reconstruct the old Raw
    # native/canonical proof graph. The fixed public query is the audit recipe.
    keys=('snapshot_id','domain','contract_id','source_profile_id','reader_version','query','derivation','limitations')
    selected={k:deepcopy(context[k]) for k in keys if k in context}
    value={'contract_version':'stock_label_price_view_v1','context':selected,
        'records_ref':digest(records),'field_meta_ref':digest(meta)}
    value['price_view_ref']=digest(value)
    return value


def _query(spec,cutoff,days):
    from axiom_data import QuerySpec
    calendar=spec['calendar']; allowed=[d for d in calendar if d<=_instant(cutoff).date().isoformat()]
    require(bool(allowed),'no actual label calendar at cutoff'); anchor=allowed[-1]
    positions={d:i for i,d in enumerate(calendar)}; endpoints={anchor}
    for day in days:
        pos=positions[day]
        from .stock_target_spec import label_horizon
        for offset in (1,label_horizon(spec)):
            if pos+offset<len(calendar) and calendar[pos+offset]<=anchor: endpoints.add(calendar[pos+offset])
    query=QuerySpec(domain='market_daily',fields=('open','close'),symbols=tuple(spec['universe']),
        sessions=tuple(sorted(endpoints)),pit_policy=spec['pit_policy'],
        cutoff_by_session={d:cutoff for d in sorted(endpoints)},purpose='label_outcomes',price_basis='unadjusted')
    return query,anchor


def _price_view(data,spec,cutoff,days,metrics):
    """Select one full public cutoff domain, never relabel sliced Data wires."""
    from .labels import _query_context
    from .data_adapter import _versioned
    from axiom_data import adjust_prices
    query,anchor=_query(spec,cutoff,days)
    query_wire={f.name:(dict(value) if isinstance(value,Mapping) else list(value) if isinstance(value,tuple) else value)
        for f in dataclass_fields(query) for value in (getattr(query,f.name),)}
    # Data owns its internal read allocations. Bound requested cells before
    # calling it, then charge the actual adjusted wire once at this boundary.
    _working(metrics,len(query.sessions)*len(spec['universe'])*4096+len(days)*len(spec['universe'])*1024)
    price=data.read(snapshot=spec['snapshot'],query=query)
    factors=data.read(snapshot=spec['snapshot'],query=replace(query,domain='adjustment_factors',fields=('factor',)))
    metrics['data_read_calls']+=2
    adjusted=adjust_prices(price,factors,fields=('open','close'),anchor_session=anchor,
                          decision_session=anchor,factor_field='factor')
    # The public adjuster returns a detached batch. Native inputs need not
    # overlap its JSON conversion, canonical hashes or graph accounting.
    del price,factors
    records,meta,context=_versioned(adjusted,'label_outcomes')
    del adjusted
    _working(metrics,2*_measured(metrics,[records,meta,context])+len(days)*len(spec['universe'])*1024)
    parsed=_query_context(context,spec['calendar']); source=_source_view(records,meta,context)
    # to_json returned detached private Data values. The opaque handle exposes
    # only copied source controls and detached Raw output rows, never these maps.
    domain=_RawPriceDomain(_RAW_DOMAIN_TOKEN,{'pid':os.getpid(),'borrowers':0,'metrics':metrics,
        'source':source,'binding':(context['snapshot_id'],context['query']['pit_policy'],
                                  tuple(spec['calendar']),parsed[1]),
        'features':set(days),'parsed':parsed,'records':records,'field_meta':meta,'compiled':None})
    records=meta=context=None
    try: domain._charge()
    except BaseException:
        domain.close(); raise
    metrics['price_domains'].append({'cutoff':cutoff,'query_ref':digest(query_wire),
        'query':query_wire,'price_view_ref':source['price_view_ref'],'data_read_calls':2,
        'resident_bytes':metrics['_price_view_bytes']})
    return domain


def _raw(spec,cutoff,days,cache,metrics,price_view):
    from .labels import _forward_rows
    from .stock_column_inputs import ColumnPriceDomain
    columnar=type(price_view) is ColumnPriceDomain
    require(type(price_view) in (_RawPriceDomain,ColumnPriceDomain),'owner-loaded Raw price domain required')
    if not columnar:price_view.require_request(spec,cutoff,days)
    else:require(price_view.spec==spec and set(days)<=set(spec['feature_sessions']),'column Raw request mismatch')
    from .stock_target_spec import label_horizon
    h=label_horizon(spec)
    source=price_view.source
    definition={'price_view':source,'snapshot':spec['snapshot'],'cutoff':cutoff,
        'calendar':spec['calendar'],'universe':spec['universe'],'sessions':days,
        'formula':f'close(f+{h}) / open(f+1) - 1','price_basis':'common_anchor_adjusted_v1',
        'horizon_sessions':h,'start_session_offset':1,'end_session_offset':h,
        'missing_policy':'invalid_null_preserve_grid','implementation_ref':_column_raw_implementation() if columnar else _raw_implementation()}
    if columnar:definition['label_definition_ref']=spec['target_spec']['label_definition_ref']
    key=digest(definition); target=(Path(cache)/'raw'/key[7:]).resolve(); artifact=target/'target.json'
    if artifact.exists():
        from .stock_compact_batch import read_target
        store=metrics['_store']; descriptor={'path':str(artifact.absolute())}
        value,rows=read_target(store,descriptor,expected=definition)
        _working(metrics,len(rows)*2048); rows=list(rows)
        metrics['raw_cache_hits']+=1
        return {**descriptor,'file_digest':store.hashes[descriptor['path']],'target_ref':value['target_ref']},rows
    if columnar:
        from axiom_engine.core import execute_forward_returns
        rows=list(price_view.rows(days,horizon=h,forward_operator=execute_forward_returns,
            reuse=metrics['_column_targets'],role=metrics.get('_column_role','training')))
    else:
        rows=list(_forward_rows(None,None,None,calendar=spec['calendar'],features=days,
            horizon_sessions=h,source_ref=source['price_view_ref'],_domain=price_view))
    charge=_measured(metrics,rows)
    _working(metrics,charge)
    metrics['raw_operator_calls']+=1
    previous=metrics['_retained_raw_bytes']
    try:
        # The writer and its saved-target byte validation run while both this
        # child and its source domain remain alive. Charge those external roots
        # to the same store, not just to the producer's separate peak counter.
        metrics['_retained_raw_bytes']=previous+charge
        _working(metrics,len(rows)*1024+65536)
        descriptor=_publish(target,definition,rows,budgets=metrics['_limits'],store=metrics['_store'])
    except BaseException:
        rows=None
        raise
    finally:
        metrics['_retained_raw_bytes']=previous
        _sync_feature_charge(metrics)
    return descriptor,rows


def _normalized(raw_parts,rows,feature,cutoff,days,cache,metrics,*,target_spec=None):
    try:
        from axiom_engine.core import execute_cs_zscore_batch
        from axiom_engine._implementation import IMPLEMENTATION_REF
        fd=_view_data(feature); spec=fd['definition']['spec']; width=len(spec['universe'])
        feature_ref=fd['definition']['feature_view_ref']
        positions={d:i for i,d in enumerate(spec['feature_sessions'])}
        offsets=[positions[d]*width+i for d in days for i in range(width)]
        _working(metrics,len(rows)*4096+len(spec['ordered_features'])*96+3072)
        reasons=[]; instant=_instant(cutoff); cutoff_us=instant_us(cutoff)
        selection=metrics.get('_model_feature_selection')
        eligibility_common=spec if target_spec is None else {**spec,'target_spec':target_spec}
        from .stock_compact_store import model_feature_binding,model_eligibility_ref
        binding=model_feature_binding(feature,selection)
        feature_count=len(spec['ordered_features']) if binding is None else len(binding['ordered_features'])
        for row,facts in zip(rows,iter_feature_eligibility(feature,offsets,model_feature_selection=selection)):
            from .stock_target_spec import eligible_target_reason
            reason=eligible_target_reason(row,facts,feature_count,instant,eligibility_common)
            if reason is None:
                require(facts.maximum_available_at_utc_us is not None and
                        facts.maximum_available_at_utc_us<=cutoff_us,
                        'training Feature native clock exceeds fit')
            reasons.append(reason)
        cohort={'contract_version':'stock_compact_cohort_v1','feature_view_ref':feature_ref,'cutoff':cutoff,
            'sessions':days,'universe':spec['universe'],'raw_refs':[d['target_ref'] for d in raw_parts],
            'eligibility_reasons':reasons,'eligible_keys':[[r['security_id'],r['feature_session']]
                for r,reason in zip(rows,reasons) if reason is None]}
        if binding is not None: cohort['model_feature_eligibility_ref']=model_eligibility_ref(binding)
        cohort_ref=digest(cohort)
        definition={'calendar':spec['calendar'],'universe':spec['universe'],'sessions':days,'cutoff':cutoff,
            'raw_refs':cohort['raw_refs'],'cohort_ref':cohort_ref,'normalization_spec':NORMALIZATION_SPEC,
            'implementation_ref':_implementation(),'core_implementation_ref':IMPLEMENTATION_REF}
        if metrics.get('_column_targets') is not None:
            numerical_implementation=_column_normalization_implementation()
            definition.update(implementation_ref=numerical_implementation,
                core_implementation_ref=numerical_implementation)
        key=digest(definition); target=(Path(cache)/'normalized'/key[7:]).resolve(); artifact=target/'target.json'
        if artifact.exists():
            from .stock_compact_batch import read_target
            store=metrics['_store']; desc={'path':str(artifact.absolute())}
            value,normalized=read_target(store,desc,expected=None if metrics.get('_column_targets') is not None else definition,raw_rows=rows)
            if metrics.get('_column_targets') is not None:
                require({k:v for k,v in value['definition'].items() if k not in
                    ('normalization_proof_version','numerical_origins')}==definition,
                    'normalized column logical request mismatch')
            metrics['normalized_cache_hits']+=1
            fd['store'].check()
            return {**desc,'file_digest':store.hashes[desc['path']],'target_ref':value['target_ref']},list(normalized),value['core_ref'],cohort
        reuse=metrics.get('_column_targets');daily={};core_days=days
        if reuse is not None:
            import numpy as np
            core_days=[]
            for i,day in enumerate(days):
                _working(metrics,width*32+4096)
                values=np.asarray([rows[i*width+j]['return'] if reasons[i*width+j] is None else 0.0
                    for j in range(width)],dtype='<f8')
                members=np.asarray([reasons[i*width+j] is None for j in range(width)],dtype='?')
                values.flags.writeable=members.flags.writeable=False
                numeric=digest({'vectors':reuse.vector_ref((values,members)),
                    'normalization_spec_ref':digest(NORMALIZATION_SPEC)})
                cached=reuse.normalization(day,numeric)
                daily[day]={'input_numeric_ref':numeric,'cached':cached}
                if cached is None:core_days.append(day)
                else:metrics['normalized_daily_reuses']=metrics.get('normalized_daily_reuses',0)+1
            values=members=cached=None
        day_positions={day:i for i,day in enumerate(days)}
        arrays={name:array(code) for name,code in {'values':'d','value_validity':'B','value_reason_codes':'i',
            'fact_available_at_utc_us':'q','reference_member':'B','reference_available_at_utc_us':'q',
            'selection_cutoff_utc_us':'q','fact_source_codes':'i','reference_source_codes':'i'}.items()}
        dictionary=[None,*sorted({r for r in reasons if r is not None})]; codes={r:i for i,r in enumerate(dictionary)}
        sources={}; calendar_ref=digest({'contract_version':'stock_label_calendar_v1','sessions':spec['calendar']})
        raw_ref=digest(cohort['raw_refs'])
        _working(metrics,_measured(metrics,cohort)+len(rows)*4096)
        for day in core_days:
            i=day_positions[day]
            eligible=[[security,day] for j,security in enumerate(spec['universe']) if reasons[i*width+j] is None]
            bindings=_normalization_sources(raw_ref,feature_ref,day,cutoff,eligible)
            sources[day]={'bindings':sorted(bindings,key=lambda r:r['id']),
                          'source_sets':[['offline_eligibility'],['raw_labels']]}
            arrays['selection_cutoff_utc_us'].append(instant_us(core_clock(cutoff)))
            for j in range(width):
                n=i*width+j; row=rows[n]; valid=reasons[n] is None
                arrays['values'].append(row['return'] if valid else 0.0); arrays['value_validity'].append(valid)
                arrays['value_reason_codes'].append(codes[reasons[n]])
                arrays['fact_available_at_utc_us'].append(instant_us(core_clock(row['label_available_at'] if valid else cutoff)))
                arrays['reference_member'].append(valid); arrays['reference_available_at_utc_us'].append(instant_us(core_clock(cutoff)))
                arrays['fact_source_codes'].append(1 if valid else 0); arrays['reference_source_codes'].append(0)
        # Core accepts immutable readonly buffer views; its arithmetic remains the
        # existing operator, not a Research vectorized substitute.
        carrier={'contract_version':'core_cs_zscore_batch_input_v1','calendar_ref':calendar_ref,
            'schema':RAW_TARGET_SCHEMA,'output_schema':NORMALIZED_TARGET_SCHEMA,'sessions':core_days,
            'security_ids':spec['universe'],'reason_dictionary':dictionary,'source_bindings_by_session':sources,
            **{k:memoryview(v.tobytes()).cast('?' if k in ('value_validity','reference_member') else v.typecode)
               for k,v in arrays.items()}}
        # Carrier bytes and per-session bindings are detached from these working
        # panels. Release them before Core takes its own immutable snapshot and
        # before normalized rows/cohort publication are allocated.
        del arrays
        try:
            result=execute_cs_zscore_batch(carrier,params=NORMALIZATION_SPEC['params']) if core_days else None
        finally:
            # A retained exception traceback must not turn the producer frame into
            # an owner of the detached Core input buffers.
            del carrier,sources,dictionary
        if core_days:
            metrics['core_calls']+=1; metrics['label_core_calls']+=1
            if reuse is not None:metrics['normalization_core_calls']=metrics.get('normalization_core_calls',0)+1
        normalized=[]
        if reuse is not None:
            import numpy as np
            for i,day in enumerate(core_days):
                # Keep only numerical outputs and their real Core origin; current
                # availability/source bindings remain a separate logical layer.
                _working(metrics,width*18+4096)
                vals=np.frombuffer(result['values'][i*width:(i+1)*width].tobytes(),dtype='<f8')
                flags=np.frombuffer(result['value_validity'][i*width:(i+1)*width].tobytes(),dtype='?')
                daily[day]['cached']=reuse.remember_normalization(day,daily[day]['input_numeric_ref'],
                    vals,flags,result['metadata']['result_ref'])
            origins=[{'session':day,'input_numeric_ref':daily[day]['input_numeric_ref'],
                'output_numeric_ref':reuse.vector_ref((daily[day]['cached']['values'],daily[day]['cached']['validity']))}
                for day in days]
            if core_days:
                metrics.setdefault('normalization_core_receipts',[]).append({
                    'core_result_ref':result['metadata']['result_ref'],
                    'sessions':list(core_days),'numerical_bindings':[r for r in origins if r['session'] in core_days]})
            proof={'contract_version':'stock_normalization_reuse_v1','normalization_spec_ref':digest(NORMALIZATION_SPEC),
                'numerical_origins':origins}
            core_ref=digest(proof)
            definition.update(normalization_proof_version='stock_normalization_reuse_v1',numerical_origins=origins)
            # The declared logical cutoff and original Core dependency projection
            # set valid target clocks; cached numerical vectors carry no old clock.
            normalized_clock=core_clock(cutoff)
        else:core_ref=result['metadata']['result_ref']
        for i,row in enumerate(rows):
            cached=None if reuse is None else daily[days[i//width]]['cached']
            valid=bool(result['value_validity'][i] if cached is None else cached['validity'][i%width])
            normalized.append({**row,'raw_return':row['return'],'raw_available_at':row['label_available_at'],
                'return':float(result['values'][i] if cached is None else cached['values'][i%width]) if valid else None,'valid':valid,
                'label_available_at':(_clock(result['available_at_utc_us'][i]) if cached is None else normalized_clock) if valid else None,
                'invalid_reason':None if valid else reasons[i] or 'NORMALIZATION_UNDEFINED'})
        # Neither the Core carrier nor its immutable result is a next-fold cache.
        # Keep only the logical normalized rows and small committed references.
        del result
        # The directory is keyed by the logical request. New v2's core_ref is an
        # explicitly versioned deterministic numerical proof, not a legacy Core
        # execution ref. Actual Core invocation refs are producer receipts and do
        # not make logical output identity depend on cache warmness/batch packing.
        desc=_publish(target,definition,normalized,budgets=metrics['_limits'],normalized=True,core_ref=core_ref,cohort=cohort,
                      store=metrics['_store'])
        fd['store'].check()
        return desc,normalized,core_ref,cohort

    finally:
        daily=values=members=cached=result=vals=flags=arrays=carrier=sources=dictionary=fd=feature=metrics=rows=raw_parts=normalized=cohort=reasons=spec=eligibility_common=None


def _prepare_legacy_compact_batch(data, *, feature_inputs,fold_specs,destination,preparation_options,metrics=None,progress=None,
                          _caller_bytes=0,_caller_source_bytes=0,model_feature_selection=None,reuse_raw_from_batch=None):
    from .stock_compact_batch import load_compact_state
    begin=time.perf_counter(); own=type(feature_inputs) is not StockFeatureView
    require(not own or isinstance(feature_inputs,(str,Path)),'Feature input must be an owner handle or path')
    options=deepcopy(preparation_options)
    required={'row_block_sessions','column_block','maximum_resident_bytes','normalization_backend'}
    require(type(options) is dict and required<=set(options)<=required|{'maximum_source_bytes','maximum_parent_bytes','control_layout'} and
            options['normalization_backend']=='core_cs_batch_v1' and
            options.get('control_layout','fold_controls_v1') in ('fold_controls_v1','inline_v3') and
            all(type(v) is int and v>0 for k,v in options.items() if k not in ('normalization_backend','control_layout')),
            'fixed compact options and positive budgets required')
    compact=options.get('control_layout','fold_controls_v1')=='fold_controls_v1'
    require(model_feature_selection is None or compact,'model_feature_selection requires compact v4 controls')
    budgets=limits({'maximum_matrix_bytes':options['maximum_resident_bytes'],
        **{k:options[k] for k in ('maximum_source_bytes','maximum_parent_bytes') if k in options}})
    require(type(_caller_bytes) is int and _caller_bytes>=0 and type(_caller_source_bytes) is int and _caller_source_bytes>=0,
            'nonnegative owner audit accounting required')
    feature=load_stock_feature_view(feature_inputs,limits=budgets,residency='sequential') if own else feature_inputs
    store=None
    try:
        from .stock_compact_store import model_feature_binding,model_common
        fd=_view_data(feature); source_spec=fd['definition']['spec']; spec=source_spec; width=len(spec['universe'])
        model_binding=model_feature_binding(feature,model_feature_selection)
        if model_binding is not None:
            spec={**source_spec,'ordered_features':model_binding['ordered_features'],'feature_selection':model_binding['selection']}
        require(fd['store'].resident_bytes<=budgets['maximum_matrix_bytes'] and
                fd['store'].metrics['source_bytes']<=budgets['maximum_source_bytes'] and
                fd['store'].metrics['largest_parent_bytes']<=budgets['maximum_parent_bytes'],'borrowed Feature budget incompatible')
        definition={'version':'axiom.stock_ml_batch_inputs/4' if compact else 'axiom.stock_ml_batch_inputs/3','feature_view':feature.to_dict(),
            'fold_specs':deepcopy(fold_specs),'preparation_options':options,'implementation_ref':_implementation(),
            'environment':_input_environment()}
        if compact: definition['price_domain_plan']='fit_window_evaluation_calendar_blocks_v1'
        if model_binding is not None: definition['model_feature_selection']=model_binding
        require(reuse_raw_from_batch is None or compact,'Raw reuse requires compact v4 controls')
        origin=None
        if reuse_raw_from_batch is not None:
            from .stock_batch import _data
            owner=_data(reuse_raw_from_batch)
            require(owner['manifest']['contract_version']=='stock_ml_batch_inputs_v4',
                    'Raw reuse requires an owner-loaded compact v4 batch')
            origin=owner['matrix_state']; origin.check()
            require(origin.compact and origin.active==0 and
                    origin.batch['definition']['feature_view']==feature.to_dict(),'Raw reuse Feature identity mismatch')
            require(origin.batch['definition']['preparation_options']['row_block_sessions']==options['row_block_sessions'],
                    'Raw reuse row-block query plan mismatch')
            require(origin.batch['definition']['fold_specs']==fold_specs,
                    'Raw reuse fold/cutoff collection mismatch')
            require(origin.batch['definition']['price_domain_plan']==definition['price_domain_plan'],
                    'Raw reuse price-domain query plan mismatch')
        definition_ref=digest(definition); target=Path(destination).absolute()/definition_ref[7:]
        reused,reused_source=_raw_reuse_charge(origin,fd['store'])
        store=OwnedStore(budgets,shared_bytes=fd['store'].resident_bytes+_caller_bytes+reused,
            shared_source_bytes=fd['store'].metrics['source_bytes']+_caller_source_bytes+reused_source)
        store.check_hook=fd['store'].check
        store.reserve(0)
    except BaseException:
        if store is not None: store.close()
        if own: feature.close()
        raise
    transferred=False
    stats={'cache_hit':False,'data_read_calls':0,'supplier_calls':0,'feature_core_calls':0,'core_calls':0,
        'label_core_calls':0,'raw_operator_calls':0,'raw_cache_hits':0,'normalized_cache_hits':0,
        'admitted_price_view_reuses':0,'account_calls':0,'train_calls':0,'predict_calls':0,'fold_queries':[],
        'price_domains':[],
        'maximum_working_bytes':fd['store'].resident_bytes,
        '_limits':budgets,'_feature_bytes':fd['store'].resident_bytes+_caller_bytes,'_retained_raw_bytes':0,'_store':store,
        '_feature_store':fd['store'],'_caller_bytes':_caller_bytes,'_caller_source_bytes':_caller_source_bytes,
        '_model_feature_selection':model_feature_selection,'_raw_origin':origin}
    try:
        _sync_feature_charge(stats)
        if (target/'batch.json').exists():
            value=json.loads((target/'batch.json').read_bytes()); require(value['definition']==definition,'cached compact definition mismatch')
            state=load_compact_state(value,feature_inputs=feature,limits=budgets,
                publication_store=store,residency='sequential')
            try: state.verify_all()
            except BaseException:
                state.close(); raise
            if value['batch_ref'] in fd['prepared']: fd['prepared'][value['batch_ref']].close()
            fd['prepared'][value['batch_ref']]=state
            transferred=True
            stats.update(cache_hit=True,total_seconds=time.perf_counter()-begin)
            if metrics is not None: metrics.update({k:v for k,v in stats.items() if not k.startswith('_')})
            return value
        require(type(fold_specs) is list and bool(fold_specs),'explicit ordered folds required')
        plans=[]; previous=None
        for fold in fold_specs:
            training,inference=validate_spec(fold,spec['calendar'])
            require((previous is None or fold['oos_trade_sessions'][0]>previous) and
                    set(training+inference)<=set(spec['feature_sessions']),'fold grid/chronology mismatch')
            previous=fold['oos_trade_sessions'][-1]; plans.append((fold,training,inference))
        cache=Path(destination).absolute()/'compact-cache'; records=[]; domains={}; raw_outputs=[]
        reused=None
        if origin is not None:
            by_spec={digest(f['fold_spec']):f for f in origin.batch['folds']}
            reused=[]
            for fold,_,_ in plans:
                require(digest(fold) in by_spec,'Raw reuse fold/cutoff mismatch')
                saved=by_spec[digest(fold)]
                source_feature=_view_data(origin.feature,check=False)['store']
                previous=(origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes)
                try:
                    origin.store.limits={k:min(v,budgets[k]) for k,v in previous[0].items()}
                    origin._fixed_shared_bytes=previous[1]+_caller_bytes+store.resident_bytes+store.lease_bytes
                    origin._fixed_shared_source_bytes=previous[2]+_caller_source_bytes+store.metrics['source_bytes']
                    if source_feature is not fd['store']:
                        origin._fixed_shared_bytes+=fd['store'].resident_bytes+fd['store'].lease_bytes
                        origin._fixed_shared_source_bytes+=fd['store'].metrics['source_bytes']
                    origin._sync_shared(); origin.store.reserve(0)
                    control=origin._parts(saved['input_manifest'],fold)
                finally:
                    origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes=previous
                    origin._sync_shared()
                _sync_feature_charge(stats)
                reused.append({'raw_parts':deepcopy(control['raw_parts']),
                    'evaluation_parts':deepcopy(control['evaluation_parts'])})
            origin.check()
            stats['raw_reused_from_batch']=origin._batch_ref
        for i,(fold,training,inference) in enumerate(plans):
            if reused is not None:
                raw_outputs.append(reused[i]); stats['raw_cache_hits']+=len(reused[i]['raw_parts'])+len(reused[i]['evaluation_parts'])
                continue
            chunks=[training[n:n+options['row_block_sessions']] for n in range(0,len(training),options['row_block_sessions'])]
            raw_outputs.append({'raw_parts':[None]*len(chunks),
                **({'evaluation_parts':[]} if compact else {'evaluation':None})})
            stats['fold_queries'].append({'fold_spec_ref':digest(fold),'data_read_calls':0,
                'admitted_price_view_reuses':0,'price_domain_refs':[]})
            jobs=[('raw_parts',n,days,fold['fit_cutoff']) for n,days in enumerate(chunks)]
            if compact:
                positions={day:n for n,day in enumerate(spec['calendar'])}; evaluation={}
                for day in inference:
                    evaluation.setdefault(positions[day]//options['row_block_sessions'],[]).append(day)
                raw_outputs[i]['evaluation_parts']=[None]*len(evaluation)
                jobs.extend(('evaluation_parts',n,days,fold['evaluation_cutoff'])
                    for n,days in enumerate(evaluation.values()))
            else: jobs.append(('evaluation',None,inference,fold['evaluation_cutoff']))
            for role,n,days,cutoff in jobs:
                if not compact: key=_instant(cutoff)
                elif role=='evaluation_parts': key=(_instant(cutoff),'evaluation',positions[days[0]]//options['row_block_sessions'])
                else: key=(_instant(cutoff),'training')
                group=domains.setdefault(key,{'cutoff':cutoff,'days':set(),'jobs':[]})
                group['days'].update(days); group['jobs'].append((i,role,n,days,cutoff))
        # One full fit domain or fixed evaluation block; release it before
        # selecting the next domain. Raw children retain the full query
        # identity, never an invented sliced DataBatch contract.
        for group in domains.values():
            stats['_price_view_bytes']=0
            with _price_view(data,spec,group['cutoff'],sorted(group['days']),stats) as price_view:
                for use,(i,role,n,days,cutoff) in enumerate(group['jobs']):
                    report=stats['fold_queries'][i]; domain_ref=price_view.source_ref
                    if domain_ref not in report['price_domain_refs']: report['price_domain_refs'].append(domain_ref)
                    if use: stats['admitted_price_view_reuses']+=1; report['admitted_price_view_reuses']+=1
                    else: report['data_read_calls']+=2
                    desc,chunk=_raw(spec,cutoff,days,cache,stats,price_view)
                    if role in ('raw_parts','evaluation_parts'): raw_outputs[i][role][n]=desc
                    else: raw_outputs[i][role]=desc
                    del chunk
                    # Published descriptors, rather than every historical typed
                    # target, are the retained union between cutoff domains.
                    store.release_payloads()
            del price_view
        from .stock_compact_batch import read_target
        positions={d:i for i,d in enumerate(spec['feature_sessions'])}
        for i,(fold,training,inference) in enumerate(plans):
            window=[positions[d]*width+j for d in training+inference for j in range(width)]
            _producer_feature_window(feature,window,stats)
            parts=raw_outputs[i]['raw_parts']; rows=[]
            for desc in parts:
                # The previous chunk changed live Python roots after its last
                # measurement. The next target read must see that new charge.
                _sync_feature_charge(stats)
                raw_value,raw_rows=read_target(store,desc); chunk=list(raw_rows)
                chunk_bytes=_measured(stats,chunk); _working(stats,chunk_bytes)
                stats['_retained_raw_bytes']+=chunk_bytes; rows.extend(chunk)
                del raw_value,raw_rows,chunk
                store.release_payloads()
            norm,nrows,core_ref,cohort=_normalized(parts,rows,feature,fold['fit_cutoff'],training,cache,stats)
            normalized_bytes=_measured(stats,[nrows,cohort]); _working(stats,normalized_bytes)
            stats['_retained_raw_bytes']+=normalized_bytes
            store.release_payloads()
            training_offsets=[positions[r['feature_session']]*width+spec['universe'].index(r['security_id'])
                for r in nrows if r['valid']]
            inference_offsets=[positions[d]*width+i for d in inference for i in range(width)]
            _working(stats,len(spec['ordered_features'])*96+3072)
            def joined_rows():
                selected=(r for r in nrows if r['valid'])
                for f,r in zip(iter_feature_rows(feature,training_offsets,
                        model_feature_selection=model_feature_selection),selected):
                    f.update(label=r['return'],raw_return=r['raw_return'],label_available_at=r['raw_available_at'],
                             normalized_available_at=r['label_available_at'])
                    yield f
            training_binding={'training_rows_ref':digest_array_rows(joined_rows()),'training_row_count':len(training_offsets),
                'training_keys_digest':digest_array_rows([spec['universe'][off%width],
                    spec['feature_sessions'][off//width]] for off in training_offsets)}
            record={'fold_spec':deepcopy(fold),'raw_parts':parts,'normalized':norm,
                'core_ref':core_ref,'cohort_ref':digest(cohort),'training_binding':training_binding}
            if compact:
                from .stock_compact_controls import selectors_for
                record.update(contract_version='stock_ml_fold_control_v1',evaluation_parts=raw_outputs[i]['evaluation_parts'])
                if model_binding is not None: record['model_feature_selection_ref']=model_binding['model_feature_selection_ref']
                selectors=selectors_for(record,spec,fd['row_index'],training_offsets)
                record=seal(record,'fold_control_ref')
                descriptor=write_part(cache/'fold-controls',record,'fold_control_ref')
                records.append({'fold_spec':record['fold_spec'],'fold_control':descriptor,'selectors':selectors,'core_ref':core_ref})
                del record,selectors,training_offsets,inference_offsets
            else:
                record.update(evaluation=raw_outputs[i]['evaluation'],training_offsets=training_offsets,inference_offsets=inference_offsets)
                records.append(record)
            if progress: progress({'stage':'compact_labels','completed':len(records),'total':len(plans)})
            # Completed fold facts live in typed targets; release their Python
            # working panels before the next cutoff-specific Data selection.
            del rows,nrows,cohort,joined_rows,window
            stats['_retained_raw_bytes']=0
            store.release_payloads()
            # Preserve already verified overlap for the next full window;
            # set_feature_window trims only partitions outside that window.
        if model_binding is None: _producer_feature_window(feature,[],stats)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.compact-batch-',dir=target.parent) as temporary:
            stage=Path(temporary)/'complete'; stage.mkdir()
            common=model_common(source_spec,model_binding)
            view=seal({'contract_version':'stock_ml_prepared_view_v3' if compact else 'stock_ml_prepared_view_v2',
                'definition':common,'feature_view':feature.to_dict(),**({} if compact else {'fold_targets':records})},'prepared_view_ref')
            write_json(stage/'view.json',view)
            view_desc={'path':str(target/'view.json'),'file_digest':file_digest(stage/'view.json'),
                       'prepared_view_ref':view['prepared_view_ref']}
            folds=[]
            for record in records:
                selectors=record['selectors'] if compact else {k:None if k=='validation' else record['training_offsets' if k in
                    ('training','training_labels') else 'inference_offsets'] for k in
                    ('training','validation','inference','training_labels','evaluation_labels')}
                inputs=seal({'contract_version':'stock_ml_saved_inputs_v4' if compact else 'stock_ml_saved_inputs_v3','prepared_view':view_desc,
                    **({'fold_control':record['fold_control']} if compact else {}),
                    'fold_spec_ref':digest(record['fold_spec']),'selectors':selectors,
                    'core_result_refs':[record['core_ref']]},'input_ref')
                if model_binding is not None:
                    inputs=seal({**{k:v for k,v in inputs.items() if k!='input_ref'},
                        'model_feature_selection_ref':model_binding['model_feature_selection_ref']},'input_ref')
                folds.append({'input_manifest':inputs,'fold_spec':record['fold_spec']})
            batch={'contract_version':'stock_ml_batch_inputs_v4' if compact else 'stock_ml_batch_inputs_v3','definition':definition,'definition_ref':definition_ref,
                'prepared_view':view_desc,'folds':folds,'status':'COMPLETE'}
            batch['batch_ref']=digest(batch); batch=seal(batch,'content_digest'); write_json(stage/'batch.json',batch)
            _sync_feature_charge(stats)
            state=load_compact_state(batch,feature_inputs=feature,limits=budgets,
                resolver=lambda p:stage/Path(p).relative_to(target) if Path(p).is_relative_to(target) else Path(p),
                publication_store=store,residency='sequential')
            # COMPLETE requires the same full saved closure validation, one
            # active fold at a time, before the directory becomes visible.
            try:
                state.verify_all()
                stage.rename(target); state.store.resolve=lambda p:Path(p); state.check()
            except BaseException:
                state.close(); raise
            fd['prepared'][batch['batch_ref']]=state
            transferred=True
        stats.update(total_seconds=time.perf_counter()-begin,batch_ref=batch['batch_ref'],
            feature_file_hash_calls=fd['store'].metrics['file_hash_calls'],
            label_file_hash_calls=store.metrics['file_hash_calls'],label_hash_bytes=store.metrics['hash_bytes'],
            label_released_buffer_bytes=store.metrics.get('released_buffer_bytes',0),
            legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        if metrics is not None: metrics.update({k:v for k,v in stats.items() if not k.startswith('_')})
        return batch
    finally:
        if transferred:
            # Raw owners are borrowed for this operation. Returned state owns
            # its label leaves and retains only the declared caller baseline.
            state._fixed_shared_bytes=_caller_bytes
            state._fixed_shared_source_bytes=_caller_source_bytes
            state._sync_shared()
        else: store.close()
        if own: feature.close()
        if stats['data_read_calls'] and callable(getattr(data,'clear_cache',None)): data.clear_cache()


class _IncrementalPreparation:
    """The existing v4 producer, paused only between verified complete folds."""
    def __init__(self,data,*,feature_inputs,fold_specs,destination,preparation_options,
                 metrics=None,progress=None,model_feature_selection=None,reuse_raw_from_batch=None,
                 _caller_bytes=0,_caller_source_bytes=0,_saved_definition=None,
                 label_spec=None,column_source=None):
        self.pid=os.getpid();self.closed=False;self.feature=self.store=self.state=self.batch=None
        self.transferred=False;self.own=type(feature_inputs) is not StockFeatureView
        self.data=data;self.metrics=metrics;self.progress=progress;self.begin=time.perf_counter()
        self.cursor=0;self.readonly=_saved_definition is not None;self.checkpoint_mark=None
        self.stats={};self.groups={};self.plans=[];self.raw_outputs=[];self.ready=[]
        self.columnar=(label_spec is not None or column_source is not None or
            (_saved_definition is not None and _saved_definition['version']=='axiom.stock_ml_batch_inputs/5'))
        self.column_source=column_source;self.column_targets=None
        self.column_previous={}
        self.options=deepcopy(preparation_options)
        if self.columnar:self.options.setdefault('control_layout','fold_controls_v2')
        required={'row_block_sessions','column_block','maximum_resident_bytes','normalization_backend'}
        require(type(self.options) is dict and required<=set(self.options)<=required|{
            'maximum_source_bytes','maximum_parent_bytes','control_layout'} and
            self.options['normalization_backend']=='core_cs_batch_v1' and
            self.options.get('control_layout','fold_controls_v1')==('fold_controls_v2' if self.columnar else 'fold_controls_v1') and
            all(type(v) is int and v>0 for k,v in self.options.items() if k not in ('normalization_backend','control_layout')),
            'fixed incremental compact v4 options and positive budgets required')
        require(type(_caller_bytes) is int and _caller_bytes>=0 and type(_caller_source_bytes) is int and _caller_source_bytes>=0,
                'nonnegative owner audit accounting required')
        self.budgets=limits({'maximum_matrix_bytes':self.options['maximum_resident_bytes'],
            **{k:self.options[k] for k in ('maximum_source_bytes','maximum_parent_bytes') if k in self.options}})
        try:
            from .stock_compact_store import model_feature_binding,model_common
            from .stock_compact_batch import _load_checkpoint_state,load_compact_state
            from .stock_batch import _compact_batch_handle
            require(not self.own or isinstance(feature_inputs,(str,Path)),'Feature input must be an owner handle or path')
            self.feature=load_stock_feature_view(feature_inputs,limits=self.budgets,residency='sequential') if self.own else feature_inputs
            fd=_view_data(self.feature);source=fd['definition']['spec']
            self.binding=model_feature_binding(self.feature,model_feature_selection)
            self.spec=source if self.binding is None else {**source,'ordered_features':self.binding['ordered_features'],
                'feature_selection':self.binding['selection']}
            if self.columnar:
                from .stock_target_spec import resolve_stock_label_spec
                target=(deepcopy(_saved_definition['target_spec']) if _saved_definition is not None
                        else resolve_stock_label_spec(label_spec))
                require(target==resolve_stock_label_spec(target['label_spec']), 'saved stock target profile mismatch')
                self.spec={**self.spec,'target_spec':target}
                require(self.readonly or column_source is not None,'shared public column source required for v5 preparation')
            self.width=len(self.spec['universe']);self.positions={d:i for i,d in enumerate(self.spec['feature_sessions'])}
            require(fd['store'].resident_bytes<=self.budgets['maximum_matrix_bytes'] and
                fd['store'].metrics['source_bytes']<=self.budgets['maximum_source_bytes'] and
                fd['store'].metrics['largest_parent_bytes']<=self.budgets['maximum_parent_bytes'],'borrowed Feature budget incompatible')
            self.definition={'version':'axiom.stock_ml_batch_inputs/4','feature_view':self.feature.to_dict(),
                'fold_specs':deepcopy(fold_specs),'preparation_options':self.options,'implementation_ref':_implementation(),
                'environment':_input_environment(),'price_domain_plan':'fit_window_evaluation_calendar_blocks_v1'}
            if self.binding is not None:self.definition['model_feature_selection']=self.binding
            if self.columnar:
                self.definition.update(version='axiom.stock_ml_batch_inputs/5',target_spec=target,
                    price_domain_plan='column_asof_endpoint_dependencies_v1',implementation_ref=_implementation(columnar=True))
            if _saved_definition is not None:
                require(all(_saved_definition[k]==self.definition[k] for k in self.definition if k not in
                    ('implementation_ref','environment')),'checkpoint saved source/options mismatch')
                self.definition=deepcopy(_saved_definition)
            self.definition_ref=digest(self.definition)
            self.target=Path(destination).absolute()/self.definition_ref[7:];self.cache=Path(destination).absolute()/'compact-cache'
            self.checkpoint=self.target/'checkpoint.json'
            require(type(fold_specs) is list and bool(fold_specs),'explicit ordered folds required')
            previous=None;calendar_positions={d:n for n,d in enumerate(self.spec['calendar'])}
            for i,fold in enumerate(self.definition['fold_specs']):
                training,inference=validate_spec(fold,self.spec['calendar'])
                require((previous is None or fold['oos_trade_sessions'][0]>previous) and
                    set(training+inference)<=set(self.spec['feature_sessions']),'fold grid/chronology mismatch')
                previous=fold['oos_trade_sessions'][-1];self.plans.append((fold,training,inference))
                chunks=[training[n:n+self.options['row_block_sessions']] for n in range(0,len(training),self.options['row_block_sessions'])]
                evaluation={}
                for day in inference:evaluation.setdefault(calendar_positions[day]//self.options['row_block_sessions'],[]).append(day)
                self.raw_outputs.append({'raw_parts':[None]*len(chunks),'evaluation_parts':[None]*len(evaluation)})
                jobs=[('raw_parts',n,days,fold['fit_cutoff'],None) for n,days in enumerate(chunks)]
                jobs.extend(('evaluation_parts',n,days,fold['evaluation_cutoff'],block) for n,(block,days) in enumerate(evaluation.items()))
                for role,n,days,cutoff,block in jobs:
                    key=(_instant(cutoff).isoformat(),role,block)
                    group=self.groups.setdefault(key,{'cutoff':cutoff,'days':set(),'jobs':[]})
                    group['days'].update(days);group['jobs'].append((i,role,n,days,cutoff))
            origin=None
            if reuse_raw_from_batch is not None:
                from .stock_batch import _data
                origin=_data(reuse_raw_from_batch)['matrix_state'];origin.check()
                require(origin.compact and origin.active==0 and origin.batch['definition']['feature_view']==self.feature.to_dict() and
                    origin.batch['definition']['fold_specs']==self.definition['fold_specs'] and
                    origin.batch['definition']['preparation_options']['row_block_sessions']==self.options['row_block_sessions'] and
                    origin.batch['definition']['price_domain_plan']==self.definition['price_domain_plan'],
                    'Raw reuse source/fold/query plan mismatch')
                if self.columnar:require(origin.batch['definition'].get('target_spec')==self.definition['target_spec'],
                    'Raw reuse source Label definition mismatch')
            reused,reused_source=_raw_reuse_charge(origin,fd['store'])
            self.store=OwnedStore(self.budgets,shared_bytes=fd['store'].resident_bytes+_caller_bytes+reused,
                shared_source_bytes=fd['store'].metrics['source_bytes']+_caller_source_bytes+reused_source)
            self.store.check_hook=fd['store'].check;self.store.reserve(0)
            self.stats={'cache_hit':False,'data_read_calls':0,'supplier_calls':0,'feature_core_calls':0,'core_calls':0,
                'label_core_calls':0,'raw_operator_calls':0,'raw_cache_hits':0,'normalized_cache_hits':0,
                'admitted_price_view_reuses':0,'account_calls':0,'train_calls':0,'predict_calls':0,'fold_queries':[
                    {'fold_spec_ref':digest(f),'data_read_calls':0,'admitted_price_view_reuses':0,'price_domain_refs':[]} for f,_,_ in self.plans],
                'price_domains':[],'maximum_working_bytes':fd['store'].resident_bytes,
                '_limits':self.budgets,'_feature_bytes':fd['store'].resident_bytes+_caller_bytes,'_retained_raw_bytes':0,
                '_store':self.store,'_feature_store':fd['store'],'_caller_bytes':_caller_bytes,'_caller_source_bytes':_caller_source_bytes,
                '_model_feature_selection':model_feature_selection,'_raw_origin':origin,'_checkpoint_bytes':0}
            self._charge_controls()
            if (self.target/'batch.json').exists():
                manifest=self.store.read_json({'path':str(self.target/'batch.json')},key='content_digest')
                require(manifest['definition']==self.definition,'cached compact definition mismatch')
                self.state=load_compact_state(manifest,feature_inputs=self.feature,limits=self.budgets,
                    publication_store=self.store,residency='sequential')
                self.state.verify_all();self.ready=self.state.batch['folds'];self.stats['cache_hit']=True
            else:
                require(not self.readonly or self.checkpoint.is_file(),'saved checkpoint required')
                self.target.mkdir(parents=True,exist_ok=True)
                common=model_common(source,self.binding)
                if self.columnar:common.update(target_spec=self.spec['target_spec'],source_contract='data_column_selection_v1')
                view=seal({'contract_version':'stock_ml_prepared_view_v4' if self.columnar else 'stock_ml_prepared_view_v3',
                    'definition':common,
                    'feature_view':self.feature.to_dict()},'prepared_view_ref')
                self._immutable_json(self.target/'view.json',view)
                self._immutable_json(self.target/'definition.json',self.definition)
                self.view_descriptor={'path':str(self.target/'view.json'),'file_digest':file_digest(self.target/'view.json'),
                    'prepared_view_ref':view['prepared_view_ref']}
                plan_descriptor={'path':str(self.target/'definition.json'),'file_digest':file_digest(self.target/'definition.json')}
                if self.checkpoint.exists():
                    self._read_checkpoint()
                self.state=_load_checkpoint_state(self.definition,self.view_descriptor,plan_descriptor,self.ready,
                    feature_inputs=self.feature,limits=self.budgets,publication_store=self.store)
                self.ready=self.state.batch['folds'];self._validate_saved_raw()
            self.batch=_compact_batch_handle(self.state,self.state.batch,self.begin)
            self.stats['_owner_state']=self.state
            self.stats['_owner_shared_baseline']=self.state._fixed_shared_bytes
            if self.columnar:
                from .stock_column_inputs import ColumnMathReuse
                self.column_targets=ColumnMathReuse(self.stats)
                self.stats['_column_targets']=self.column_targets
            self._charge_controls()
        except BaseException:
            self.close();raise
        finally:
            fd=source=view=manifest=origin=fold_specs=preparation_options=_saved_definition=None
            training=inference=jobs=chunks=evaluation=calendar_positions=group=key=days=fold=None
            feature_inputs=data=metrics=progress=model_feature_selection=reuse_raw_from_batch=None

    def _check(self):
        require(self.pid==os.getpid(),'checkpoint producer belongs to another process')
        require(not self.closed,'checkpoint producer is closed')
        self.state.check()

    def _immutable_json(self,path,value):
        try:
            if path.exists():
                require(self.store.read_json({'path':str(path)})==value,'checkpoint immutable source changed');return
            require(not self.readonly,'checkpoint immutable source missing')
            with tempfile.TemporaryDirectory(prefix='.checkpoint-control-',dir=self.target) as temporary:
                stage=Path(temporary)/'control.json';write_json(stage,value)
                try:os.link(stage,path)
                except FileExistsError:
                    require(self.store.read_json({'path':str(path)})==value,'concurrent checkpoint source conflict')
        finally:value=None

    def _charge_controls(self,*,temporary_bytes=0):
        external=0 if self.state is None else max(0,self.state._fixed_shared_bytes-self.stats['_owner_shared_baseline'])
        amount=_size([self.definition,self.plans,self.groups,self.raw_outputs,self.ready,self.positions,
            {k:v for k,v in self.stats.items() if not k.startswith('_')}],maximum=self.store.maximum_matrix_bytes,
            retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes+temporary_bytes)
        self.store.reserve(max(0,amount-self.stats['_checkpoint_bytes'])+temporary_bytes)
        self.stats['_checkpoint_bytes']=amount;_sync_feature_charge(self.stats)
        if self.state is None:return
        baseline=self.stats['_caller_bytes']+self.stats.get('raw_reuse_live_bytes',0)+amount+self.stats.get('_column_math_bytes',0)
        self.state._fixed_shared_bytes=baseline+external
        self.stats['_owner_shared_baseline']=baseline
        self.state._fixed_shared_source_bytes=self.store.shared_source_bytes-_view_data(self.feature)['store'].metrics['source_bytes']
        self.state._sync_shared()

    def _read_checkpoint(self):
        """Mutable operational bytes get a bounded temporary admission owner."""
        _sync_feature_charge(self.stats)
        temporary=OwnedStore(self.budgets,shared_bytes=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes,
            shared_source_bytes=self.store.shared_source_bytes+self.store.metrics['source_bytes'])
        temporary.check_hook=self.stats['_feature_store'].check
        saved=outputs=actual=expected=None
        try:
            saved=temporary.read_json({'path':str(self.checkpoint)},key='checkpoint_ref')
            require(set(saved)=={'definition_ref','prepared_view','raw_outputs','folds','checkpoint_ref'} and
                saved['definition_ref']==self.definition_ref and saved['prepared_view']==self.view_descriptor,
                'checkpoint final plan/fields mismatch')
            outputs=saved['raw_outputs']
            require(type(outputs) is list and len(outputs)==len(self.raw_outputs),'checkpoint Raw coverage mismatch')
            for actual,expected in zip(outputs,self.raw_outputs):
                require(type(actual) is dict and set(actual)==set(expected) and all(type(actual[k]) is list and
                    len(actual[k])==len(expected[k]) for k in expected),'checkpoint Raw layout mismatch')
            self.ready=saved['folds'];self.raw_outputs=outputs
            self._charge_controls(temporary_bytes=temporary.resident_bytes+temporary.lease_bytes)
            temporary.check();self.checkpoint_mark=temporary.marks[str(self.checkpoint)]
        finally:
            saved=outputs=actual=expected=None;temporary.close();temporary=None

    def _release_payloads(self):
        self.store.release_payloads(keep_paths=self.state._keep_paths)

    def _validate_saved_raw(self):
        from .stock_compact_batch import read_target,_raw_binding
        from .stock_compact_controls import validate_price_part
        desc=value=rows=definition=None
        try:
            for key,group in self.groups.items():
                present=[self.raw_outputs[i][role][n] is not None for i,role,n,_,_ in group['jobs']]
                require(not any(present) or all(present),'unfinished checkpoint price domain cannot HIT')
                if not all(present):continue
                for i,role,n,days,cutoff in group['jobs']:
                    desc=self.raw_outputs[i][role][n];value,rows=read_target(self.store,desc)
                    definition=value['definition']
                    require(value['contract_version']==('stock_compact_raw_v2' if self.columnar else 'stock_compact_raw_v1') and definition['sessions']==days,
                        'checkpoint Raw child mismatch')
                    _raw_binding(definition,self.state.view['definition'],cutoff)
                    if not self.columnar:
                        validate_price_part(definition,self.state.view['definition'],self.state.price_domains,
                            self.options['row_block_sessions'],evaluation=role=='evaluation_parts')
                    desc=value=rows=definition=None;self._release_payloads()
            for fold,outputs in zip(self.ready,self.raw_outputs):
                record=self.state._parts(fold['input_manifest'],fold['fold_spec'])
                require(record['raw_parts']==outputs['raw_parts'] and record['evaluation_parts']==outputs['evaluation_parts'],
                    'checkpoint fold/Raw control mismatch')
                record=None;self.state._release_window(keep_feature=True)
        finally:
            desc=value=rows=definition=record=group=None

    def _persist(self):
        from .stock_fold_inputs import file_fingerprint
        if self.readonly:return
        if self.checkpoint_mark is not None:
            require(file_fingerprint(self.checkpoint)==self.checkpoint_mark,'checkpoint changed during producer lifetime')
        value=None
        try:
            value=seal({'definition_ref':self.definition_ref,'prepared_view':self.state.batch['prepared_view'],
                'raw_outputs':self.raw_outputs,'folds':self.ready},'checkpoint_ref')
            with tempfile.TemporaryDirectory(prefix='.checkpoint-save-',dir=self.target) as temporary:
                stage=Path(temporary)/'checkpoint.json';write_json(stage,value);os.replace(stage,self.checkpoint)
            self.checkpoint_mark=file_fingerprint(self.checkpoint)
        finally:value=None

    def _ensure_raw(self,index):
        origin=self.stats['_raw_origin'];control=price_view=chunk=desc=None
        try:
            for group in self.groups.values():
                if not any(i==index for i,_,_,_,_ in group['jobs']):continue
                if all(self.raw_outputs[i][role][n] is not None for i,role,n,_,_ in group['jobs']):
                    self.stats['raw_cache_hits']+=sum(i==index for i,_,_,_,_ in group['jobs']);continue
                if origin is not None:
                    self._reuse_raw_group(group)
                    self._charge_controls()
                    continue
                require(self.column_source is not None if self.columnar else self.data is not None,'checkpoint fold is not prepared')
                self.stats['_price_view_bytes']=0
                factory=self._column_price_view if self.columnar else lambda cutoff,days,role:_price_view(self.data,self.spec,cutoff,days,self.stats)
                role=group['jobs'][0][1]
                with factory(group['cutoff'],sorted(group['days']),role) as price_view:
                    for use,(i,role,n,days,cutoff) in enumerate(group['jobs']):
                        report=self.stats['fold_queries'][i];ref=price_view.source_ref
                        if ref not in report['price_domain_refs']:report['price_domain_refs'].append(ref)
                        if use:self.stats['admitted_price_view_reuses']+=1;report['admitted_price_view_reuses']+=1
                        else:report['data_read_calls']+=2
                        desc,chunk=_raw(self.spec,cutoff,days,self.cache,self.stats,price_view)
                        self.raw_outputs[i][role][n]=desc;chunk=None;self._release_payloads()
                price_view=None
            self._charge_controls()
        finally:origin=control=price_view=chunk=desc=saved=group=None

    def _column_price_view(self,cutoff,days,role):
        """Call only Data's public selection and common-anchor adjustment API."""
        from .stock_column_inputs import ColumnPriceDomain
        query,anchor=_query(self.spec,cutoff,days);source=self.column_source
        previous=self.column_previous.get(role,{})
        cells=len(query.sessions)*len(query.symbols)
        _working(self.stats,cells*160+65536)
        prices=factors=adjusted=domain=None
        try:
            prices=source.select(query=query,previous=previous.get('prices'))
            factors=source.select(query=replace(query,domain='adjustment_factors',fields=('factor',)),
                previous=previous.get('factors'))
            adjusted=source.adjust(prices,factors,fields=('open','close'),anchor_session=anchor,
                decision_session=anchor,factor_field='factor',previous=previous.get('adjusted'))
            self.stats['data_read_calls']+=2
            domain=ColumnPriceDomain(adjusted,spec=self.spec,
                query=replace(query,price_basis='common_anchor_adjusted_v1',adjustment_anchor=anchor),anchor=anchor,
                input_queries={'price':prices.query_binding,'factor':factors.query_binding},metrics=self.stats)
            self.column_previous[role]={'prices':prices,'factors':factors,'adjusted':adjusted}
            for old in previous.values():old.close()
            self.stats['_column_role']=role
            self.stats['_price_view_bytes']=sum(a.nbytes for panel in domain.panels.values()
                for a in panel.values() if hasattr(a,'nbytes'))+sum(a.nbytes for a in domain.lineage.values())+_size([domain.source,domain.positions])
            _sync_feature_charge(self.stats);_working(self.stats,0)
            return domain
        except BaseException:
            if domain is not None:domain.close()
            for selected in (adjusted,factors,prices):
                if selected is not None:selected.close()
            raise
        finally:prices=factors=adjusted=domain=selected=old=None

    def _reuse_raw_group(self,group):
        """Reuse every child of the touched final domain under both live budgets."""
        origin=self.stats['_raw_origin'];control=saved=desc=source_feature=fstore=None
        try:
            require(origin.active==0,'Raw reuse source is borrowed');origin.check()
            for i,role,n,_,_ in group['jobs']:
                if self.raw_outputs[i][role][n] is not None:continue
                saved=origin.batch['folds'][i]
                fstore=self.stats['_feature_store'];source_feature=_view_data(origin.feature,check=False)['store']
                previous=(origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes)
                try:
                    _sync_feature_charge(self.stats)
                    external=self.stats['_feature_bytes']-fstore.resident_bytes-self.stats['raw_reuse_live_bytes']
                    origin.store.limits={k:min(v,self.budgets[k]) for k,v in previous[0].items()}
                    origin._fixed_shared_bytes=previous[1]+external+self.store.resident_bytes+self.store.lease_bytes
                    origin._fixed_shared_source_bytes=previous[2]+self.stats['_caller_source_bytes']+self.store.metrics['source_bytes']
                    if source_feature is not fstore:
                        origin._fixed_shared_bytes+=fstore.resident_bytes+fstore.lease_bytes
                        origin._fixed_shared_source_bytes+=fstore.metrics['source_bytes']
                    origin._sync_shared();origin.store.reserve(0)
                    control=origin._parts(saved['input_manifest'],saved['fold_spec'])
                    desc=deepcopy(control[role][n])
                finally:
                    origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes=previous
                    origin._sync_shared();control=None
                self.raw_outputs[i][role][n]=desc;desc=None
                self.stats['raw_cache_hits']+=1
                _sync_feature_charge(self.stats)
            origin.check();self.stats['raw_reused_from_batch']=origin._batch_ref
        finally:origin=control=saved=desc=source_feature=fstore=group=None

    def next_fold(self):
        from .stock_compact_batch import read_target
        from .stock_compact_controls import selectors_for
        from .stock_batch import _data
        self._check()
        if self.cursor==len(self.plans):return None
        index=self.cursor
        if index<len(self.ready):
            require(digest(self.ready[index]['fold_spec']) in self.state._verified_folds,
                    'unfinished checkpoint fold cannot HIT')
            self.cursor+=1;return deepcopy(self.ready[index])
        require(not self.readonly,'checkpoint fold is not prepared')
        fold,training,inference=self.plans[index];rows=[];nrows=cohort=chunk=raw_rows=raw_value=record=selectors=window=joined_rows=None
        try:
            self._ensure_raw(index)
            window=[self.positions[d]*self.width+j for d in training+inference for j in range(self.width)]
            _producer_feature_window(self.feature,window,self.stats);parts=self.raw_outputs[index]['raw_parts']
            for desc in parts:
                _sync_feature_charge(self.stats);raw_value,raw_rows=read_target(self.store,desc);chunk=list(raw_rows)
                charge=_measured(self.stats,chunk);_working(self.stats,charge);self.stats['_retained_raw_bytes']+=charge
                rows.extend(chunk);raw_value=raw_rows=chunk=None;self._release_payloads()
            norm,nrows,core_ref,cohort=_normalized(parts,rows,self.feature,fold['fit_cutoff'],training,self.cache,self.stats,
                target_spec=self.spec.get('target_spec'))
            charge=_measured(self.stats,[nrows,cohort]);_working(self.stats,charge);self.stats['_retained_raw_bytes']+=charge
            self._release_payloads()
            security_positions={security:i for i,security in enumerate(self.spec['universe'])}
            offsets=[self.positions[r['feature_session']]*self.width+security_positions[r['security_id']] for r in nrows if r['valid']]
            def joined_rows():
                for f,r in zip(iter_feature_rows(self.feature,offsets,model_feature_selection=self.stats['_model_feature_selection']),
                    (r for r in nrows if r['valid'])):
                    f.update(label=r['return'],raw_return=r['raw_return'],label_available_at=r['raw_available_at'],
                        normalized_available_at=r['label_available_at']);yield f
            binding={'training_rows_ref':None if self.columnar else digest_array_rows(joined_rows()),'training_row_count':len(offsets),
                'training_keys_digest':digest_array_rows([self.spec['universe'][off%self.width],self.spec['feature_sessions'][off//self.width]] for off in offsets)}
            record={'contract_version':'stock_ml_fold_control_v1','fold_spec':deepcopy(fold),'raw_parts':parts,'normalized':norm,
                'evaluation_parts':self.raw_outputs[index]['evaluation_parts'],'core_ref':core_ref,'cohort_ref':digest(cohort),'training_binding':binding}
            if self.binding is not None:record['model_feature_selection_ref']=self.binding['model_feature_selection_ref']
            selectors=selectors_for(record,self.spec,_view_data(self.feature)['row_index'],offsets)
            if self.columnar:
                from .stock_training_blocks import training_block_binding
                record['contract_version']='stock_ml_fold_control_v2'
                record['training_binding']=training_block_binding(self.feature,offsets,normalized=norm,
                    cohort_ref=digest(cohort),selector=selectors['training'],store=self.store,
                    destination=self.cache/'feature-proofs',model_feature_selection=self.stats['_model_feature_selection'],
                    metrics=self.stats)
            record=seal(record,'fold_control_ref');descriptor=write_part(self.cache/'fold-controls',record,'fold_control_ref')
            inputs={'contract_version':'stock_ml_saved_inputs_v5' if self.columnar else 'stock_ml_saved_inputs_v4',
                'prepared_view':self.state.batch['prepared_view'],'fold_control':descriptor,
                'fold_spec_ref':digest(fold),'selectors':selectors,'core_result_refs':[core_ref]}
            if self.binding is not None:inputs['model_feature_selection_ref']=self.binding['model_feature_selection_ref']
            item={'input_manifest':seal(inputs,'input_ref'),'fold_spec':deepcopy(fold)}
            rows=nrows=cohort=chunk=raw_rows=raw_value=joined_rows=None;self.stats['_retained_raw_bytes']=0
            _sync_feature_charge(self.stats)
            self.state._append_checkpoint_fold(item)
            value=_data(self.batch);value['fold_keys'].add((digest(item['input_manifest']),digest(item['fold_spec'])))
            self._charge_controls();self._persist();self.cursor+=1
            if self.progress:self.progress({'stage':'compact_labels','completed':len(self.ready),'total':len(self.plans)})
            return deepcopy(item)
        finally:
            rows=nrows=cohort=chunk=raw_rows=raw_value=record=selectors=window=joined_rows=parts=offsets=norm=descriptor=inputs=item=security_positions=None
            self.stats['_retained_raw_bytes']=0
            _sync_feature_charge(self.stats)

    def finish(self):
        from .stock_batch import _data
        self._check()
        if not self.state.incomplete:return self.batch.to_dict()
        require(len(self.ready)==len(self.plans) and self.cursor==len(self.plans),'complete checkpoint coverage required')
        manifest=None
        try:
            manifest={'contract_version':'stock_ml_batch_inputs_v5' if self.columnar else 'stock_ml_batch_inputs_v4',
                'definition':self.definition,'definition_ref':self.definition_ref,
                'prepared_view':self.state.batch['prepared_view'],'folds':self.ready,'status':'COMPLETE'}
            manifest['batch_ref']=digest(manifest);manifest=seal(manifest,'content_digest')
            require(self.state._verified_folds=={digest(f) for f,_,_ in self.plans},
                    'complete verified checkpoint coverage required')
            self.state.check()
            self.state._complete_checkpoint(manifest,publish=lambda:self._immutable_json(self.target/'batch.json',manifest))
            value=_data(self.batch);value['manifest']=manifest;value['identity']=manifest['batch_ref'];value['incomplete']=False
            self.stats.update(batch_ref=manifest['batch_ref'],total_seconds=time.perf_counter()-self.begin)
            return self.batch.to_dict()
        finally:manifest=None

    def transfer(self):
        from .stock_batch import _data
        require(not self.own and not self.state.incomplete,'complete borrowed Feature transfer required')
        fd=_view_data(self.feature);ref=self.state._batch_ref
        if ref in fd['prepared']:fd['prepared'][ref].close()
        fd['prepared'][ref]=self.state;_data(self.batch).clear();_data_guard={'owner_pid':self.pid,'closed':True}
        from .stock_batch import _DATA
        _DATA[self.batch].update(_data_guard);self.transferred=True

    def close(self):
        if self.closed:return
        require(self.pid==os.getpid(),'checkpoint producer belongs to another process')
        if self.state is not None:require(self.state.active==0 and getattr(self.state,'_build_oos_borrowers',0)==0,'checkpoint owner still borrowed')
        self.closed=True
        if self.stats:
            self.stats.update(total_seconds=time.perf_counter()-self.begin,
                feature_file_hash_calls=self.stats['_feature_store'].metrics['file_hash_calls'],
                label_file_hash_calls=self.store.metrics['file_hash_calls'],label_hash_bytes=self.store.metrics['hash_bytes'],
                label_released_buffer_bytes=self.store.metrics.get('released_buffer_bytes',0),
                legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        if self.metrics is not None:self.metrics.update({k:v for k,v in self.stats.items() if not k.startswith('_')})
        caller=self.stats.get('_caller_bytes',0);caller_source=self.stats.get('_caller_source_bytes',0)
        reads=self.stats.get('data_read_calls',0)
        self.plans=self.groups=self.raw_outputs=self.ready=self.definition=self.spec=self.positions=self.binding=self.options=None
        if self.transferred:
            self.state._fixed_shared_bytes=caller;self.state._fixed_shared_source_bytes=caller_source;self.state._sync_shared()
        elif self.batch is not None:self.batch.close()
        elif self.state is not None:self.state.close()
        elif self.store is not None:self.store.close()
        if self.column_targets is not None:self.column_targets.close()
        self.stats.clear()
        if self.own and self.feature is not None:self.feature.close()
        # ColumnSource is caller-owned and shared across preparation/model/OOS.
        # Clearing Data here would revoke that external source and its borrows.
        if not self.columnar and reads and self.data is not None and callable(getattr(self.data,'clear_cache',None)):self.data.clear_cache()
        self.batch=self.state=self.store=self.feature=self.data=None
        for previous in self.column_previous.values():
            for selection in previous.values():selection.close()
        self.column_previous.clear()
        self.column_source=self.column_targets=None


@contextmanager
def _prepare_compact_incrementally(data,**kwargs):
    owner=None
    try:
        owner=_IncrementalPreparation(data,**kwargs)
        kwargs.clear();data=None
        yield owner
    finally:
        kwargs.clear();data=None
        if owner is not None:owner.close()
        owner=None


@contextmanager
def _load_compact_checkpoint_owner(path,*,feature_inputs=None):
    """Read only a private saved checkpoint; no Data or numerical stage."""
    admission=OwnedStore();definition=saved=source=selection=None
    try:
        definition=admission.read_json({'path':str(Path(path).absolute()/'definition.json')})
        options=definition['preparation_options']
        admission.limits=limits({'maximum_matrix_bytes':options['maximum_resident_bytes'],
            **{k:options[k] for k in ('maximum_source_bytes','maximum_parent_bytes') if k in options}})
        admission.reserve(0)
        require(admission.metrics['source_bytes']<=admission.limits['maximum_source_bytes'] and
            admission.metrics['largest_parent_bytes']<=admission.limits['maximum_parent_bytes'],'checkpoint plan byte budget exceeded')
        saved=definition['feature_view']
        source=feature_inputs if feature_inputs is not None else Path(saved['source_index']['path']).parent
        selection=definition.get('model_feature_selection')
        extra=admission.resident_bytes+admission.lease_bytes;extra_source=admission.metrics['source_bytes']
        with _prepare_compact_incrementally(None,feature_inputs=source,fold_specs=definition['fold_specs'],
            destination=Path(path).parent,preparation_options=options,_caller_bytes=extra,_caller_source_bytes=extra_source,
            model_feature_selection=None if selection is None else selection['selection'],_saved_definition=definition) as owner:
            definition=saved=source=selection=options=None;admission.close()
            owner.stats['_caller_bytes']-=extra;owner.stats['_caller_source_bytes']-=extra_source;owner._charge_controls()
            yield owner.batch
    finally:
        definition=saved=source=selection=options=feature_inputs=None;admission.close();admission=owner=None


def prepare_compact_batch(data,*,feature_inputs,fold_specs,destination,preparation_options,metrics=None,progress=None,
                          _caller_bytes=0,_caller_source_bytes=0,model_feature_selection=None,reuse_raw_from_batch=None,
                          label_spec=None,column_source=None):
    require(type(preparation_options) is dict,'fixed compact options required')
    if preparation_options.get('control_layout','fold_controls_v1')=='inline_v3':
        require(label_spec is None and column_source is None,'column targets require the explicit v5 controls')
        return _prepare_legacy_compact_batch(data,feature_inputs=feature_inputs,fold_specs=fold_specs,destination=destination,
            preparation_options=preparation_options,metrics=metrics,progress=progress,_caller_bytes=_caller_bytes,
            _caller_source_bytes=_caller_source_bytes,model_feature_selection=model_feature_selection,reuse_raw_from_batch=reuse_raw_from_batch)
    with _prepare_compact_incrementally(data,feature_inputs=feature_inputs,fold_specs=fold_specs,destination=destination,
        preparation_options=preparation_options,metrics=metrics,progress=progress,_caller_bytes=_caller_bytes,
        _caller_source_bytes=_caller_source_bytes,model_feature_selection=model_feature_selection,reuse_raw_from_batch=reuse_raw_from_batch,
        label_spec=label_spec,column_source=column_source) as owner:
        while owner.next_fold() is not None:pass
        result=owner.finish()
        if not owner.own:owner.transfer()
        return result

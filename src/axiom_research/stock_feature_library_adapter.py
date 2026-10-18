"""Native Feature-library consumption by the existing training owner.

This adapter has its own saved contract. It never labels a projected library
view as a legacy Feature producer result. Existing Label/fit readers receive
their genuine StockFeatureView owner handle; cold restoration uses this loader
and passes that handle explicitly to load_stock_ml_batch_inputs.
"""
from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

from axiom_engine.feature_library import FeatureView
from axiom_engine.feature_library.codec import digest, seal
from axiom_engine.feature_library.store import VerifiedMatrixStore
from .stock_fold_inputs import require, ordered
from .stock_matrix_storage import row_index, write_buffer, write_part, write_partition
from .stock_compact_store import OwnedStore, StockFeatureView, _TOKEN, _size


VERSION='stock_feature_library_training_projection_v1'


def _feature_source_compatibility(view, source, selection, *, raw_view=None, raw_bindings=None, catalog=None):
    """Same semantics and local selected facts, allowing unchanged historical frames."""
    from axiom_engine.feature_library import feature_definitions
    from .feature_catalog import load_feature_catalog_ref
    from .stock_feature_library import normalized_feature_dependency_ref, admitted_raw_dependency_refs
    cap=view.describe(); physical=source.source; spec=physical.manifest['spec']
    require(cap['security_ids']==spec['universe'] and set(cap['sessions'])<=set(spec['feature_sessions']) and
            [s['id'] for s in selection]==cap['columns'], 'Feature/source axes or selection mismatch')
    require(catalog is None or raw_view is not None, 'explicit catalog requires saved raw binding')
    catalog=load_feature_catalog_ref(spec['catalog_ref']) if catalog is None else catalog
    catalog.select(selection)
    require(catalog.to_dict()['input_contract']=={'price_basis':'common_anchor_adjusted_v1',
        'adjustment_anchor':'feature_session_visible_at_cutoff','amount_unit':'CNY'},
        'unsupported training source/catalog input interpretation')
    if raw_view is not None or raw_bindings is not None:
        from .stock_feature_library import _saved_normalization_request, _RawCrossSectionSource
        require(raw_view is not None and raw_bindings is not None,
                'saved normalization requires both raw view and explicit raw owners')
        raw_bindings=list(raw_bindings)
        request=_saved_normalization_request(raw_view,source,catalog,selection)
        expected=feature_definitions(request.plan,input_semantics=request.input_semantics)
        raw_cap=raw_view.describe()
        require(all(cap[k]==raw_cap[k] for k in ('sessions','security_ids','sample_universe','columns')) and
                cap['definitions']==expected,'saved normalized semantics or axes mismatch')
        adapter=_RawCrossSectionSource(raw_view,source,raw_bindings=raw_bindings,catalog=catalog,selection=selection)
        dependencies=adapter.dependency_refs(expected,cap['sessions'],cap['security_ids'])
        for day,columns in dependencies.items():
            receipts=view.column_receipts(day,list(columns))
            require(all(receipts[n]['dependency_ref']==ref for n,ref in columns.items()),
                    'saved normalized raw/reference dependency mismatch')
        # Cold readers resolve the same installed immutable catalog by digest.
        load_feature_catalog_ref(catalog.identity)
        return seal({'contract_version':'stock_feature_source_compatibility_v1','layer':'normalized',
            'view_ref':view.identity,'semantic_refs':{n:expected[n]['semantic_ref'] for n in cap['columns']},
            'dependency_refs':dependencies,'reference_security_ids':list(source.reference_security_ids),
            'source':source.descriptor,'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],
            'cutoff_by_session':{day:spec['cutoff_by_session'][day] for day in cap['sessions']},
            'feature_catalog_ref':catalog.identity,
            'feature_recipe_ref':catalog.recipe_ref(selection,normalized=True),
            'member_catalog_ref':spec['catalog_ref'],
            'raw_view':raw_view.descriptor,
            'raw_bindings':[{'view':v.descriptor,'source':s.descriptor} for v,s in raw_bindings]},'compatibility_ref')
    raw=source.definitions(catalog,selection,normalized=False)
    def same(expected):
        return all(cap['definitions'][name]['semantic_ref']==expected[name]['semantic_ref'] and
                   cap['definitions'][name]['schema']==expected[name]['schema'] for name in cap['columns'])
    layer='raw' if same(raw) else None
    if layer is None:
        normalized=source.definitions(catalog,selection,normalized=True)
        layer='normalized' if same(normalized) else None
    require(layer is not None, 'Feature/source definition semantics mismatch')
    dependencies=(admitted_raw_dependency_refs(view,source,raw,cap['sessions'],cap['security_ids'])
                  if layer=='raw' else source.dependency_refs(raw,cap['sessions'],cap['security_ids']))
    if layer=='normalized':
        dependencies={day:{name:normalized_feature_dependency_ref(source,day,raw[name]['semantic_ref'],ref)
                           for name,ref in columns.items()} for day,columns in dependencies.items()}
    if layer=='normalized':
        for day,columns in dependencies.items():
            receipts=view.column_receipts(day,list(columns))
            require(all(receipts[name]['dependency_ref']==ref for name,ref in columns.items()),
                    'Feature/source local dependency or reference mismatch')
    return seal({'contract_version':'stock_feature_source_compatibility_v1','layer':layer,
        'view_ref':view.identity,'semantic_refs':{n:cap['definitions'][n]['semantic_ref'] for n in cap['columns']},
        'dependency_refs':dependencies,'reference_security_ids':list(source.reference_security_ids),
        'source':source.descriptor,'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],
        'cutoff_by_session':{day:spec['cutoff_by_session'][day] for day in cap['sessions']}},'compatibility_ref')


def _instant_us(value, null):
    value=int(value)
    if value==null: return None
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=value)).isoformat().replace('+00:00','Z')


def export_feature_library_training_projection(view, source, *, selection, destination,
                                        maximum_working_bytes=512*1024**2):
    """Export bounded daily blocks without Feature computation or new mathematics."""
    import numpy as np
    capabilities=view.describe(); physical=source.source; source.store.check()
    compatibility=_feature_source_compatibility(view,source,selection)
    spec=deepcopy(physical.manifest['spec']); days=capabilities['sessions']; securities=capabilities['sample_universe']
    spec.update(universe=securities,feature_sessions=days,feature_selection=deepcopy(selection),
                ordered_features=list(capabilities['columns']))
    last=spec['read_sessions'].index(days[-1])+1
    spec['read_sessions']=spec['read_sessions'][:last]
    spec['cutoff_by_session']={d:spec['cutoff_by_session'][d] for d in spec['read_sessions']}
    schema=[deepcopy(capabilities['definitions'][n]['schema']) for n in capabilities['columns']]
    definition={'spec':spec,'library_view':view.descriptor,'source':source.descriptor,
                'reference_security_ids':source.reference_security_ids,
                'source_compatibility':compatibility,
                'projection':'preserve_packed_cells_and_original_sample_membership_v1'}
    target=Path(destination).resolve()/digest(definition)[7:]; target.mkdir(parents=True,exist_ok=True)
    grid=row_index(days,securities); rid=write_part(target,grid,'row_index_ref')
    positions=[physical.manifest['spec']['universe'].index(s) for s in securities]; partitions=[]
    for ordinal,day in enumerate(days):
        with view.operation():
            cells=view.project([(s,day) for s in securities]); count=len(securities)
            require(count*len(schema)*512+4096<=maximum_working_bytes, 'training projection row budget exceeded')
            member=physical.arrays['member'][physical.positions[day],positions]
            clocks=cells['available_at_utc_us']; known=clocks!=cells['null_available_at_utc_us']
            arrays={'values':np.where(cells['validity'],cells['values'],0).astype('<f8'),
                    'value_validity':cells['validity'].astype('u1'),
                    'available_at_utc_us':np.where(known,clocks,0).astype('<i8'),
                    'available_at_validity':known.astype('u1')}
            buffers={name:write_buffer(target,array.reshape(-1),dtype='float64_le' if name=='values' else
                'int64_le' if name=='available_at_utc_us' else 'bool_u8',shape=list(array.shape))
                for name,array in arrays.items()}
            parent=seal({'contract_version':'stock_feature_library_daily_projection_v1','view_ref':view.identity,
                'session':day,'security_ids':securities,'columns':capabilities['columns'],
                'partition_refs':cells['partition_refs'],'buffers':buffers,
                'membership_digest':digest(member.astype('u1').tolist()),'source':source.descriptor},'projection_ref')
            rows=[{'security_id':s,'session':day,'knowledge_cutoff':spec['cutoff_by_session'][day],
                'member':bool(member[i]),'validity':[bool(v) for v in cells['validity'][i]],
                'availability':[_instant_us(v,cells['null_available_at_utc_us']) for v in clocks[i]],
                'reasons':[[] if cells['reason_dictionary'][int(code)] is None else
                           [cells['reason_dictionary'][int(code)]] for code in cells['reason_codes'][i]],
                'source_refs':[view.identity,parent['projection_ref'],*cells['partition_refs']]} for i,s in enumerate(securities)]
            metadata=seal({'contract_version':'stock_feature_library_training_metadata_v1','rows':rows,
                'projection':parent,'row_references':{day:{'feature_ref':parent['projection_ref'],
                    'qlib_view_ref':digest(physical.manifest['qlib_view'])}}},'metadata_ref')
            view.validate_boundary()
            md=write_part(target,metadata,'metadata_ref')
            partitions.append(write_partition(target,table='features',fold_spec_ref=None,row_index_ref=grid['row_index_ref'],
                schema_digest=digest(schema),row_offset=ordinal*count,row_count=count,
                columns=capabilities['columns'],buffers=buffers,metadata=md))
    source.store.check(); view.validate_boundary()
    wire=seal({'contract_version':VERSION,'definition':definition,'schema':schema,'row_index':rid,
        'source_member_buffer':deepcopy(physical.manifest['buffers']['member']),
        'source_member_sessions':physical.manifest['spec']['read_sessions'],
        'source_member_securities':physical.manifest['spec']['universe'],
        'partitions':partitions,'status':'COMPLETE'},'training_adapter_ref')
    return write_part(target,wire,'training_adapter_ref')


def load_feature_library_training_view(descriptor, *, limits=None, source=None):
    """Admit native selected partitions, or verify an explicit legacy export."""
    import numpy as np
    store=OwnedStore(limits); handle=None
    try:
        wire=store.read_json(descriptor,key='training_adapter_ref')
        require(wire['contract_version'] in (VERSION,NATIVE_VERSION) and wire['status']=='COMPLETE','complete training view required')
        if wire['contract_version']==NATIVE_VERSION:
            return _load_native_training_view(descriptor,wire,store)
        definition=wire['definition']; spec=deepcopy(definition['spec']); days=ordered(spec['feature_sessions'],'Feature days')
        securities=ordered(spec['universe'],'Feature securities'); columns=spec['ordered_features']; schema=wire['schema']
        from .stock_matrix_reader import _schema
        _schema(schema,columns); grid=store.read_json(wire['row_index'],key='row_index_ref')
        require(grid==row_index(days,securities), 'training projection row grid mismatch')
        blocks=[]; parents={}; count=len(securities)
        from .stock_feature_library import SavedColumnFeatureSource
        with (SavedColumnFeatureSource(definition['source'],reference_security_ids=definition['reference_security_ids'],
                  maximum_source_bytes=store.limits['maximum_source_bytes'],maximum_working_bytes=store.maximum_matrix_bytes)
              if source is None else nullcontext(source)) as admitted_source, \
             FeatureView(definition['library_view'],maximum_matrix_bytes=store.maximum_matrix_bytes,
                         maximum_source_bytes=store.limits['maximum_source_bytes']) as library, \
             VerifiedMatrixStore(maximum_matrix_bytes=store.maximum_matrix_bytes,
                                 maximum_source_bytes=store.limits['maximum_source_bytes']) as member_store:
            capabilities=library.describe()
            require(admitted_source.descriptor==definition['source'], 'explicit admitted training source differs from saved binding')
            require(definition['source_compatibility']==_feature_source_compatibility(
                library,admitted_source,spec['feature_selection']),
                'training Feature/source compatibility receipt mismatch')
            require(capabilities['sessions']==days and capabilities['sample_universe']==securities and
                    capabilities['columns']==columns and
                    [capabilities['definitions'][n]['schema'] for n in columns]==schema,
                    'training projection library binding mismatch')
            if wire['source_member_buffer'].get('layout') in ('date_segments_v1','security_segments_v1'):
                from .feature_input_buffers import SegmentedFeatureBuffer
                members=SegmentedFeatureBuffer(wire['source_member_buffer'],member_store)
            else:members=member_store.buffer(wire['source_member_buffer']).view('bool')
            original_source=member_store.read_json(definition['source'],'source_ref')
            original_spec=original_source['spec']
            require(wire['source_member_buffer']==original_source['buffers']['member'] and
                    wire['source_member_sessions']==original_spec['read_sessions'] and
                    wire['source_member_securities']==original_spec['universe'] and
                    all(spec[key]==original_spec[key] for key in ('snapshot','pit_policy','calendar','catalog_ref')) and
                    set(days)<=set(original_spec['feature_sessions']) and
                    all(spec['cutoff_by_session'][d]==original_spec['cutoff_by_session'][d] for d in spec['read_sessions']),
                    'training projection fixed source/membership binding mismatch')
            member_axes={s:i for i,s in enumerate(wire['source_member_securities'])}
            member_days={d:i for i,d in enumerate(wire['source_member_sessions'])}
            require(len(wire['partitions'])==len(days),'complete daily training projection required')
            for ordinal,(day,part) in enumerate(zip(days,wire['partitions'])):
                with library.operation(), store.operation(), member_store.operation():
                    store.check_path(descriptor['path']); store.check_path(wire['row_index']['path'])
                    member_store.check_path(wire['source_member_buffer']['path'])
                    member_store.check_path(definition['source']['path'])
                    require(part['partition_ref']==digest({k:v for k,v in part.items() if k!='partition_ref'}) and
                        part['row_index_ref']==grid['row_index_ref'] and part['schema_digest']==digest(schema) and
                        part['row_offset']==ordinal*count and part['row_count']==count and part['columns']==columns,
                        'training projection partition grid mismatch')
                    metadata=store.read_json(part['metadata'],key='metadata_ref'); proof=metadata['projection']
                    require(proof['projection_ref']==digest({k:v for k,v in proof.items() if k!='projection_ref'}) and
                        proof['view_ref']==library.identity and proof['session']==day and
                        proof['source']==definition['source'] and proof['buffers']==part['buffers'],
                        'training projection receipt mismatch')
                    expected=library.project([(s,day) for s in securities]); known=expected['available_at_utc_us']!=expected['null_available_at_utc_us']
                    actual={k:store.buffer(v) for k,v in part['buffers'].items()}
                    for key,values in {'values':np.where(expected['validity'],expected['values'],0),
                        'value_validity':expected['validity'],'available_at_utc_us':np.where(known,expected['available_at_utc_us'],0),
                        'available_at_validity':known}.items():
                        require(np.array_equal(actual[key],values),'training projection differs from library bytes')
                    member=members[member_days[day],[member_axes[s] for s in securities]]
                    require(proof['partition_refs']==expected['partition_refs'] and
                        proof['membership_digest']==digest(member.astype('u1').tolist()) and
                        [r['member'] for r in metadata['rows']]==member.tolist(), 'training projection membership/lineage mismatch')
                    for i,row in enumerate(metadata['rows']):
                        require((row['security_id'],row['session'])==(securities[i],day) and
                            row['knowledge_cutoff']==spec['cutoff_by_session'][day] and
                            row['validity']==expected['validity'][i].tolist() and
                            row['availability']==[_instant_us(v,expected['null_available_at_utc_us'])
                                                  for v in expected['available_at_utc_us'][i]],
                            'training projection row clocks/flags differ from library')
                        require(row['reasons']==[[] if expected['reason_dictionary'][int(c)] is None else
                            [expected['reason_dictionary'][int(c)]] for c in expected['reason_codes'][i]],
                            'training projection reasons differ from library')
                        require(row['source_refs']==[library.identity,proof['projection_ref'],*expected['partition_refs']],
                                'training projection source references differ from library')
                    store.release([part['metadata']['path'],*(v['path'] for v in part['buffers'].values())])
                    blocks.append({'start':ordinal*count,'count':count,'descriptors':[part],'parts':None,'rows':None,
                        'complete_columns':True,'charge':0,'eligibility':None})
        owner_definition={'contract_version':'stock_feature_library_training_view_v1','source_index':descriptor,
            'spec':spec,'schema':schema,'row_index':wire['row_index'],'partitions':wire['partitions'],
            'historical_feature_inputs_ref':definition['library_view']['view_ref'],
            'historical_content_digest':wire['training_adapter_ref'],'library_projection_ref':wire['training_adapter_ref']}
        owner_definition['feature_view_ref']=digest(owner_definition)
        control=_size([owner_definition,grid,blocks]); store.reserve(control); store.resident_bytes+=control
        handle=StockFeatureView(_TOKEN,{'path':Path(descriptor['path']).parent,'definition':owner_definition,
            'store':store,'blocks':blocks,'parents':parents,'row_index':grid,
            'column_positions':{c:i for i,c in enumerate(columns)},'day_blocks':list(range(len(days))),
            'day_admissions':[[i] for i in range(len(days))],'control_paths':{descriptor['path'],wire['row_index']['path']},
            'residency':'sequential','closed':False,'borrowers':0,'prepared':{}})
        store.check(); return handle
    except BaseException:
        if handle is not None: handle.close()
        else: store.close()
        raise


NATIVE_VERSION='stock_feature_library_training_native_v1'


def build_feature_library_training_view(view, source, *, selection, destination,
                                        maximum_working_bytes=512*1024**2, raw_view=None, raw_bindings=None, catalog=None):
    """Save a small native binding; no training value/metadata table is copied.

    Source compatibility is checked once at this publication boundary. Fresh
    consumers admit selected Feature partitions and membership, not ancestors.
    The explicit export_feature_library_training_projection retains the old
    interoperability export for consumers that actually need physical tables.
    For normalize_saved outputs, pass the original raw_view/raw_bindings to
    re-admit raw and member dependencies independently. source explicitly owns
    daily reference/sample membership and the training spec; raw source IDs
    and snapshots remain in the saved compatibility proof and Core lineage.
    catalog independently selects an installed immutable feature catalog; it
    does not replace the membership source catalog or evidence.
    """
    cap=view.describe(); physical=source.source
    compatibility=_feature_source_compatibility(view,source,selection,raw_view=raw_view,raw_bindings=raw_bindings,catalog=catalog)
    compatibility={**{k:v for k,v in compatibility.items()
                      if k not in ('dependency_refs','compatibility_ref')},
                   'local_dependency_ref':digest(compatibility['dependency_refs'])}
    spec=deepcopy(physical.manifest['spec'])
    spec.update(universe=cap['sample_universe'],feature_sessions=cap['sessions'],
                feature_selection=deepcopy(selection),ordered_features=cap['columns'])
    if 'feature_catalog_ref' in compatibility:
        spec['catalog_ref']=compatibility['feature_catalog_ref']
    schema=[deepcopy(cap['definitions'][n]['schema']) for n in cap['columns']]
    definition={'spec':spec,'library_view':view.descriptor,'source':source.descriptor,
                'source_compatibility':compatibility}
    target=Path(destination).resolve()/digest(definition)[7:]
    target.mkdir(parents=True,exist_ok=True)
    grid=row_index(cap['sessions'],cap['sample_universe'])
    wire=seal({'contract_version':NATIVE_VERSION,'status':'COMPLETE',
        'definition':definition,'schema':schema,'row_index':grid,
        'source_member_buffer':physical.manifest['buffers']['member'],
        'source_member_sessions':physical.manifest['spec']['read_sessions'],
        'source_member_securities':physical.manifest['spec']['universe'],
        'qlib_view_ref':digest(physical.manifest['qlib_view'])},'training_adapter_ref')
    source.store.check();view.validate_boundary()
    return write_part(target,wire,'training_adapter_ref')


class _NativeTrainingBlocks:
    """Day-local projections owned by the existing StockFeatureView lifetime."""
    def __init__(self, library, members, member_store, wire):
        self.library=library;self.members=members;self.member_store=member_store;self.wire=wire
        self.days={d:i for i,d in enumerate(wire['source_member_sessions'])}
        self.securities={s:i for i,s in enumerate(wire['source_member_securities'])}
        from .stock_feature_library import _SourceArrays
        from .stock_feature_inputs import source_universe_binding
        self.member_manifest=member_store.read_json(wire['definition']['source'],'source_ref')
        binding=source_universe_binding(self.member_manifest,reader=member_store.read_json,reuse_saved=True)
        self.member_arrays=_SourceArrays(self.member_manifest,member_store,binding['universe_id'])

    def check(self):
        self.library.validate_boundary();self.member_store.check()

    def close(self):
        self.member_arrays.clear();self.library.close();self.member_store.close();self.members=None

    def evaluation_sources(self, days, columns):
        """Reuse the admitted native partitions and membership owner records."""
        from axiom_engine.feature_library.coverage import partition_descriptor
        from axiom_engine.feature_library.library import _partition_buffer_paths
        self.check()
        library=self.library; paths={library.descriptor['path']}
        for day in days:
            for column in columns:
                for tile in library._tiles[day,column]:
                    descriptor=partition_descriptor(tile['partition'])
                    require(tile['partition']['partition_ref'] in library._loaded,
                            'evaluation native Feature partition not admitted')
                    part=library._parts[tile['partition']['partition_ref']]
                    paths.add(descriptor['path']);paths.update(_partition_buffer_paths(part))
                    wire=library._store.json[descriptor['path']]
                    if 'path' in wire['core_header']:paths.add(wire['core_header']['path'])
        records={};marks={}
        for owner,chosen in ((library._store,paths),(self.member_store,self.member_store._hashes)):
            for path in chosen:
                records[path]=owner._hashes[path];marks[path]=owner.fingerprints[path]
        return records,marks

    def admit(self, value, block, columns):
        import numpy as np
        from .stock_matrix_reader import _CompactRows
        self.check();store=value['store'];key=tuple(columns)
        cache=block.setdefault('native_cache',{})
        if key in cache:return cache[key]
        spec=value['definition']['spec'];width=len(spec['ordered_features'])
        securities=spec['universe'];count=len(securities)
        day=spec['feature_sessions'][block['start']//count]
        store.reserve(count*(len(columns)*136+256)+4096)
        cells=self.library.project([(s,day) for s in securities],columns=columns)
        known=cells['available_at_utc_us']!=cells['null_available_at_utc_us']
        arrays={'values':np.where(cells['validity'],cells['values'],0).astype('<f8'),
            'value_validity':cells['validity'].astype('u1'),
            'available_at_utc_us':np.where(known,cells['available_at_utc_us'],0).astype('<i8'),
            'available_at_validity':known.astype('u1')}
        cutoff=spec['cutoff_by_session'][day]
        from .stock_matrix_storage import instant_us
        require(bool(np.isfinite(arrays['values']).all()) and
                bool((~known|(arrays['available_at_utc_us']<=instant_us(cutoff))).all()),
                'native Feature values/availability exceed original cutoff')
        member_positions=[self.securities[s] for s in securities]
        member_day=self.days[day]
        member=self.members[member_day,member_positions]
        ma=self.member_arrays
        require(bool(ma['member_known'][member_day,member_positions].all()) and
                bool(ma['member_pool_complete'][member_day,member_positions].all()) and
                bool(ma['member_available_validity'][member_day,member_positions].all()) and
                bool((ma['member_available'][member_day,member_positions]<=instant_us(cutoff)).all()),
                'model research membership must be known, complete and visible')
        positions=[value['column_positions'][n] for n in columns]
        rows=[]
        for i,s in enumerate(securities):
            flags=[False]*len(columns);clocks=[None]*len(columns);reasons=[[] for _ in columns]
            for j,k in enumerate(range(len(columns))):
                flags[k]=bool(cells['validity'][i,j])
                clocks[k]=_instant_us(cells['available_at_utc_us'][i,j],cells['null_available_at_utc_us'])
                reason=cells['reason_dictionary'][int(cells['reason_codes'][i,j])]
                reasons[k]=[] if reason is None else [reason]
            rows.append({'member':bool(member[i]),'knowledge_cutoff':cutoff,
                'validity':flags,'availability':clocks,'reasons':reasons,
                'source_refs':[self.library.identity,*cells['partition_refs']]})
        compact=_CompactRows(rows,len(columns),np);rows=None
        compact.column_positions={original:selected for selected,original in enumerate(positions)}
        require(bool((~cells['validity']|known).all()),'valid Feature requires a source clock in Research profile')
        flags=np.any(cells['validity'],axis=1)
        maximum=np.max(cells['available_at_utc_us'],axis=1)
        flags.flags.writeable=False;maximum.flags.writeable=False
        from hashlib import sha256
        buffers={name:{'dtype':'float64_le' if name=='values' else 'int64_le' if name=='available_at_utc_us' else 'bool_u8',
                      'shape':list(a.shape),'buffer_digest':'sha256:'+sha256(a.tobytes()).hexdigest()}
                 for name,a in arrays.items()}
        part={'columns':list(columns),'metadata':self.library.descriptor,'buffers':buffers,
              'partition_ref':digest({'view_ref':self.library.identity,'day':day,'columns':list(columns),
                                      'original_partitions':cells['partition_refs']})}
        for a in arrays.values():a.flags.writeable=False
        charge=compact.bytes+sum(a.nbytes for a in arrays.values())+flags.nbytes+maximum.nbytes+_size(part)+1024
        store.reserve(charge);store.resident_bytes+=charge
        block['native_charge']=block.get('native_charge',0)+charge
        cache[key]=({part['partition_ref']:(part,arrays)},compact,(flags,maximum))
        value['parents'][day]={'feature_ref':part['partition_ref'],'qlib_view_ref':self.wire['qlib_view_ref']}
        store.metrics['native_feature_projections']=store.metrics.get('native_feature_projections',0)+1
        self.check()
        return cache[key]


def _load_native_training_view(descriptor,wire,store):
    library=member_store=None
    try:
        definition=wire['definition'];spec=definition['spec'];grid=wire['row_index']
        days=ordered(spec['feature_sessions'],'native Feature sessions')
        securities=ordered(spec['universe'],'native Feature securities');columns=spec['ordered_features']
        require(grid==row_index(days,securities),'native Feature row index mismatch')
        library=FeatureView(definition['library_view'],maximum_matrix_bytes=store.maximum_matrix_bytes,
                            maximum_source_bytes=store.limits['maximum_source_bytes'])
        cap=library.describe()
        from .stock_matrix_reader import _schema
        _schema(wire['schema'],columns)
        require(cap['sessions']==days and cap['sample_universe']==securities and cap['columns']==columns and
                wire['schema']==[cap['definitions'][n]['schema'] for n in columns],
                'native training view axes/schema mismatch')
        compatibility=definition['source_compatibility']
        require(compatibility['view_ref']==library.identity and
                compatibility['source']==definition['source'] and
                compatibility['snapshot']==spec['snapshot'] and
                compatibility['pit_policy']==spec['pit_policy'] and
                compatibility['semantic_refs']=={n:cap['definitions'][n]['semantic_ref'] for n in columns} and
                compatibility['cutoff_by_session']=={d:spec['cutoff_by_session'][d] for d in days},
                'native publication compatibility binding mismatch')
        from .stock_matrix_reader import VerifiedMatrixStore as MembershipStore
        member_store=MembershipStore(maximum_matrix_bytes=store.maximum_matrix_bytes,
                                        maximum_source_bytes=store.limits['maximum_source_bytes'])
        original=member_store.read_json(definition['source'],'source_ref')
        require(wire['source_member_buffer']==original['buffers']['member'] and
                wire['source_member_sessions']==original['spec']['read_sessions'] and
                wire['source_member_securities']==original['spec']['universe'] and
                all(spec[k]==original['spec'][k] for k in ('snapshot','pit_policy','calendar','scope')) and
                set(days)<=set(original['spec']['feature_sessions']) and
                set(securities)<=set(wire['source_member_securities']) and
                all(spec['cutoff_by_session'][d]==original['spec']['cutoff_by_session'][d] for d in days),
                'native training membership/cutoff binding mismatch')
        if 'feature_catalog_ref' in compatibility:
            from .feature_catalog import load_feature_catalog_ref
            catalog=load_feature_catalog_ref(compatibility['feature_catalog_ref'])
            require(catalog.identity==compatibility['feature_catalog_ref']==spec['catalog_ref'] and
                    compatibility['member_catalog_ref']==original['spec']['catalog_ref'],
                    'native feature/member catalog binding mismatch')
            require(catalog.recipe_ref(spec['feature_selection'],normalized=True)==compatibility['feature_recipe_ref'],
                    'native feature catalog selection mismatch')
        else:
            require(spec['catalog_ref']==original['spec']['catalog_ref'],
                    'native training catalog binding mismatch')
        if wire['source_member_buffer'].get('layout') in ('date_segments_v1','security_segments_v1'):
            from .feature_input_buffers import SegmentedFeatureBuffer
            members=SegmentedFeatureBuffer(wire['source_member_buffer'],member_store)
        else:members=member_store.buffer(wire['source_member_buffer']).view('bool')
        native=_NativeTrainingBlocks(library,members,member_store,wire)
        owner_definition={'contract_version':'stock_feature_library_training_view_v1',
            'source_index':descriptor,'spec':deepcopy(spec),'schema':wire['schema'],'row_index':grid,
            'partitions':[],'historical_feature_inputs_ref':library.identity,
            'historical_content_digest':wire['training_adapter_ref'],'native_library_view':definition['library_view']}
        owner_definition['feature_view_ref']=digest(owner_definition)
        count=len(securities)
        blocks=[{'start':i*count,'count':count,'descriptors':[],'parts':None,'rows':None,
                 'complete_columns':True,'charge':0,'eligibility':None} for i in range(len(days))]
        control=_size([owner_definition,grid,blocks]);store.reserve(control);store.resident_bytes+=control
        store.check_hook=native.check
        return StockFeatureView(_TOKEN,{'path':Path(descriptor['path']).parent,'definition':owner_definition,
            'store':store,'blocks':blocks,'parents':{},'row_index':grid,'native':native,
            'column_positions':{c:i for i,c in enumerate(columns)},'day_blocks':list(range(len(days))),
            'day_admissions':[[i] for i in range(len(days))],'control_paths':{descriptor['path']},
            'residency':'sequential','closed':False,'borrowers':0,'prepared':{}})
    except BaseException:
        if library is not None:library.close()
        if member_store is not None:member_store.close()
        raise

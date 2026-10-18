"""Native Feature-library consumption by the existing training owner.

This adapter has its own saved contract. It never labels a projected library
view as a legacy Feature producer result. Existing Label/fit readers receive
their genuine StockFeatureView owner handle; cold restoration uses this loader
and passes that handle explicitly to load_stock_ml_batch_inputs.
"""
from copy import deepcopy
from contextlib import nullcontext, contextmanager, ExitStack
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
    """Record published input identities; preparation owns their provenance."""
    cap=view.describe();spec=source.source.manifest['spec']
    require(cap['security_ids']==spec['universe'] and
            [item['id'] for item in selection]==cap['columns'], 'Feature/source axes or selection mismatch')
    layer='normalized' if all(cap['definitions'][n]['schema']['stage']=='cross_sectional' for n in cap['columns']) else 'raw'
    result={'contract_version':'stock_feature_source_compatibility_v2','layer':layer,
        'view_ref':view.identity,'semantic_refs':{n:cap['definitions'][n]['semantic_ref'] for n in cap['columns']},
        'reference_security_ids':source.reference_security_ids,'source':source.descriptor,
        'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],
        'cutoff_by_session':{day:spec['cutoff_by_session'][day] for day in cap['sessions']}}
    if raw_view is not None:result['raw_view']=raw_view.descriptor
    if raw_bindings is not None:result['raw_bindings']=[{'view':v.descriptor,'source':owner.descriptor} for v,owner in raw_bindings]
    if catalog is not None:
        result.update(feature_catalog_ref=catalog.identity,
            feature_recipe_ref=catalog.recipe_ref(selection,normalized=layer=='normalized'),member_catalog_ref=spec['catalog_ref'])
    return seal(result,'compatibility_ref')


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
        for ordinal,part in enumerate(wire['partitions']):
            require(part['row_offset']==ordinal*count and part['row_count']==count and part['columns']==columns,
                    'training projection partition grid mismatch')
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

    Publication records the selected Feature and membership references. Fresh
    consumers read those partitions without walking their ancestors.
    The explicit export_feature_library_training_projection retains the old
    interoperability export for consumers that actually need physical tables.
    For normalize_saved outputs, pass the original raw_view/raw_bindings to
    retain raw and member references for traceability. source explicitly owns
    daily reference/sample membership and the training spec; raw source IDs
    and snapshots remain in the saved compatibility proof and Core lineage.
    catalog independently selects an installed immutable feature catalog; it
    does not replace the membership source catalog or evidence.
    """
    cap=view.describe(); physical=source.source
    compatibility=_feature_source_compatibility(view,source,selection,raw_view=raw_view,raw_bindings=raw_bindings,catalog=catalog)
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


class _NativeCellRows:
    """Typed facts; materialize strings/lists only for requested OOS rows."""
    def __init__(self, member, cutoff, columns, sources):
        import numpy as np
        self.member=member; self.member.flags.writeable=False; self.columns=columns
        self.dictionary=[cutoff,sources]
        self.knowledge=np.zeros(len(member),dtype='<u4')
        self.sources=np.ones(len(member),dtype='<u4')
        self.bytes=self.knowledge.nbytes+self.sources.nbytes+member.nbytes+_size(self.dictionary)+128*len(columns)
        self.knowledge.flags.writeable=False;self.sources.flags.writeable=False

    def row(self, ordinal, *, security, session, values, validity):
        reasons=[];clocks=[]
        for item in self.columns:
            reason=item['reason_dictionary'][int(item['reason_codes'][ordinal])]
            reasons.append([] if reason is None else [reason])
            clocks.append(_instant_us(item['clocks'][ordinal],item['null_clock']))
        return {'security_id':security,'session':session,'member':bool(self.member[ordinal]),
            'knowledge_cutoff':self.dictionary[0],'source_refs':list(self.dictionary[1]),
            'values':values,'validity':validity,'reasons':reasons,'availability':clocks}


class _NativeTrainingBlocks:
    """Day-local projections owned by the existing StockFeatureView lifetime."""
    def __init__(self, library, members, member_store, wire):
        self.library=library;self.members=members;self.member_store=member_store;self.wire=wire
        self._operation_depth=0
        self.days={d:i for i,d in enumerate(wire['source_member_sessions'])}
        self.securities={s:i for i,s in enumerate(wire['source_member_securities'])}
        from .stock_feature_library import _SourceArrays
        self.member_manifest=member_store.read_json(wire['definition']['source'],'source_ref')
        binding=self.member_manifest.get('universe_binding')
        universe_id=(binding['universe_id'] if binding is not None else self.member_manifest['qlib_view']['universe_query']['universe_id'])
        self.member_arrays=_SourceArrays(self.member_manifest,member_store,universe_id)

    @contextmanager
    def operation(self):
        if not self._operation_depth:self.check()
        with ExitStack() as stack:
            stack.enter_context(self.library.operation())
            stack.enter_context(self.member_store.operation())
            self._operation_depth+=1
            try:yield self
            finally:self._operation_depth-=1

    def check(self):
        if not self._operation_depth:
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
        self.check();store=value['store'];key=tuple(columns)
        cache=block.setdefault('native_cache',{})
        if key in cache:return cache[key]
        spec=value['definition']['spec'];width=len(spec['ordered_features'])
        securities=spec['universe'];count=len(securities)
        day=spec['feature_sessions'][block['start']//count]
        store.reserve(count*(len(columns)*136+256)+4096)
        owned=block.setdefault('native_columns',{})
        missing=[name for name in columns if name not in owned]
        if missing:
            cells=self.library.project([(s,day) for s in securities],columns=missing)
            known=cells['available_at_utc_us']!=cells['null_available_at_utc_us']
            from .stock_matrix_storage import instant_us
            require(bool(np.isfinite(cells['values'][cells['validity']]).all()) and
                    bool((~cells['validity']|known).all()) and
                    bool((~known|(cells['available_at_utc_us']<=instant_us(spec['cutoff_by_session'][day]))).all()),
                    'native Feature values/availability exceed original cutoff')
            for j,name in enumerate(missing):
                arrays={'values':np.where(cells['validity'][:,j:j+1],cells['values'][:,j:j+1],0).astype('<f8'),
                    'value_validity':cells['validity'][:,j:j+1].astype('u1'),
                    'available_at_utc_us':np.where(known[:,j:j+1],cells['available_at_utc_us'][:,j:j+1],0).astype('<i8'),
                    'available_at_validity':known[:,j:j+1].astype('u1')}
                part={'columns':[name],'metadata':self.library.descriptor,'buffers':{},
                    'source_partitions':[tile['partition'] for tile in self.library._tiles[day,name]]}
                item={'part':part,'arrays':arrays,'clocks':cells['available_at_utc_us'][:,j].copy(),
                    'null_clock':cells['null_available_at_utc_us'],'reason_codes':cells['reason_codes'][:,j].copy(),
                    'reason_dictionary':cells['reason_dictionary'],'source_refs':sorted({tile['partition']['partition_ref'] for tile in self.library._tiles[day,name]})}
                for a in [*arrays.values(),item['clocks'],item['reason_codes']]:a.flags.writeable=False
                charge=sum(a.nbytes for a in arrays.values())+item['clocks'].nbytes+item['reason_codes'].nbytes+_size(part)+_size(item['reason_dictionary'])+_size(item['source_refs'])+1024
                store.reserve(charge);store.resident_bytes+=charge
                block['native_charge']=block.get('native_charge',0)+charge;owned[name]=item
            store.metrics['native_feature_column_projections']=store.metrics.get('native_feature_column_projections',0)+len(missing)
        cutoff=spec['cutoff_by_session'][day]
        from .stock_matrix_storage import instant_us
        member_positions=[self.securities[s] for s in securities]
        member_day=self.days[day]
        member=self.members[member_day,member_positions]
        ma=self.member_arrays
        require(bool(ma['member_known'][member_day,member_positions].all()) and
                bool(ma['member_pool_complete'][member_day,member_positions].all()) and
                bool(ma['member_available_validity'][member_day,member_positions].all()) and
                bool((ma['member_available'][member_day,member_positions]<=instant_us(cutoff)).all()),
                'model research membership must be known, complete and visible')
        chosen=[owned[n] for n in columns]
        sources=list(dict.fromkeys(tile['partition']['partition_ref'] for name in columns
            for tile in self.library._tiles[day,name] if set(securities)&tile['security_set']))
        compact=_NativeCellRows(member,cutoff,chosen,[self.library.identity,*sources])
        flags=np.zeros(count,dtype=bool);maximum=np.full(count,-(2**63),dtype='<i8')
        for item in chosen:
            flags |= item['arrays']['value_validity'][:,0].astype(bool)
            np.maximum(maximum,item['clocks'],out=maximum)
        flags.flags.writeable=False;maximum.flags.writeable=False
        parts={name:(item['part'],item['arrays']) for name,item in zip(columns,chosen)}
        charge=compact.bytes+flags.nbytes+maximum.nbytes+256*len(parts)
        store.reserve(charge);store.resident_bytes+=charge
        block['native_charge']=block.get('native_charge',0)+charge
        cache[key]=(parts,compact,(flags,maximum))
        value['parents'][day]={'feature_ref':digest({'view_ref':self.library.identity,'day':day,
            'columns':list(columns),'original_partitions':sources}),
                               'qlib_view_ref':self.wire['qlib_view_ref']}
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
        from .stock_matrix_reader import VerifiedMatrixStore as MembershipStore
        member_store=MembershipStore(maximum_matrix_bytes=store.maximum_matrix_bytes,
                                        maximum_source_bytes=store.limits['maximum_source_bytes'])
        # This native membership owner is private and retains immutable metadata.
        # Reuse the existing ledger instead of walking its graph per legacy chunk.
        if wire['source_member_buffer'].get('layout') in ('date_segments_v1','security_segments_v1'):
            from .feature_input_buffers import SegmentedFeatureBuffer
            members=SegmentedFeatureBuffer(wire['source_member_buffer'],member_store)
        else:members=member_store.buffer(wire['source_member_buffer']).view('bool')
        native=_NativeTrainingBlocks(library,members,member_store,wire)
        owner_definition={'contract_version':'stock_feature_library_training_view_v1',
            'source_index':descriptor,'spec':spec,'schema':wire['schema'],'row_index':grid,
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

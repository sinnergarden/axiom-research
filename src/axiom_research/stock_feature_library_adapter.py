"""Explicit packed Feature-library projection into the existing training owner.

This adapter has its own saved contract. It never labels a projected library
view as a legacy Feature producer result. Existing Label/fit readers receive
their genuine StockFeatureView owner handle; cold restoration uses this loader
and passes that handle explicitly to load_stock_ml_batch_inputs.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from axiom_engine.feature_library import FeatureView
from axiom_engine.feature_library.codec import digest, seal
from axiom_engine.feature_library.store import VerifiedMatrixStore
from .stock_fold_inputs import require, ordered
from .stock_matrix_storage import row_index, write_buffer, write_part, write_partition
from .stock_compact_store import OwnedStore, StockFeatureView, _TOKEN, _size


VERSION='stock_feature_library_training_projection_v1'


def _instant_us(value, null):
    value=int(value)
    if value==null: return None
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=value)).isoformat().replace('+00:00','Z')


def build_feature_library_training_view(view, source, *, selection, destination,
                                        maximum_working_bytes=512*1024**2):
    """Export bounded daily blocks without Feature computation or new mathematics."""
    import numpy as np
    capabilities=view.describe(); physical=source.source; source.store.check()
    from .feature_catalog import load_feature_catalog_ref
    registered=load_feature_catalog_ref(physical.manifest['spec']['catalog_ref']).select(selection)
    require([f['id'] for f in registered]==capabilities['columns'], 'unregistered training projection selection')
    require(capabilities['security_ids']==physical.manifest['spec']['universe'], 'Feature/source physical axes mismatch')
    require([s['id'] for s in selection]==capabilities['columns'], 'Feature training selection mismatch')
    spec=deepcopy(physical.manifest['spec']); days=capabilities['sessions']; securities=capabilities['sample_universe']
    spec.update(universe=securities,feature_sessions=days,feature_selection=deepcopy(selection),
                ordered_features=list(capabilities['columns']))
    last=spec['read_sessions'].index(days[-1])+1
    spec['read_sessions']=spec['read_sessions'][:last]
    spec['cutoff_by_session']={d:spec['cutoff_by_session'][d] for d in spec['read_sessions']}
    schema=[deepcopy(capabilities['definitions'][n]['schema']) for n in capabilities['columns']]
    definition={'spec':spec,'library_view':view.descriptor,'source':source.descriptor,
                'reference_security_ids':source.reference_security_ids,
                'projection':'preserve_packed_cells_and_original_sample_membership_v1'}
    target=Path(destination).resolve()/digest(definition)[7:]; target.mkdir(parents=True,exist_ok=True)
    grid=row_index(days,securities); rid=write_part(target,grid,'row_index_ref')
    positions=[physical.manifest['spec']['universe'].index(s) for s in securities]; partitions=[]
    for ordinal,day in enumerate(days):
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
        md=write_part(target,metadata,'metadata_ref')
        partitions.append(write_partition(target,table='features',fold_spec_ref=None,row_index_ref=grid['row_index_ref'],
            schema_digest=digest(schema),row_offset=ordinal*count,row_count=count,
            columns=capabilities['columns'],buffers=buffers,metadata=md))
    source.store.check(); view.describe()
    wire=seal({'contract_version':VERSION,'definition':definition,'schema':schema,'row_index':rid,
        'source_member_buffer':deepcopy(physical.manifest['buffers']['member']),
        'source_member_sessions':physical.manifest['spec']['read_sessions'],
        'source_member_securities':physical.manifest['spec']['universe'],
        'partitions':partitions,'status':'COMPLETE'},'training_adapter_ref')
    return write_part(target,wire,'training_adapter_ref')


def load_feature_library_training_view(descriptor, *, limits=None):
    """Fresh byte admission + exact library comparison; reuse existing row/fit owner."""
    import numpy as np
    store=OwnedStore(limits); handle=None
    try:
        wire=store.read_json(descriptor,key='training_adapter_ref')
        require(wire['contract_version']==VERSION and wire['status']=='COMPLETE','complete training projection required')
        definition=wire['definition']; spec=deepcopy(definition['spec']); days=ordered(spec['feature_sessions'],'Feature days')
        securities=ordered(spec['universe'],'Feature securities'); columns=spec['ordered_features']; schema=wire['schema']
        from .stock_matrix_reader import _schema
        _schema(schema,columns); grid=store.read_json(wire['row_index'],key='row_index_ref')
        require(grid==row_index(days,securities), 'training projection row grid mismatch')
        blocks=[]; parents={}; count=len(securities)
        with FeatureView(definition['library_view'],maximum_matrix_bytes=store.maximum_matrix_bytes,
                         maximum_source_bytes=store.limits['maximum_source_bytes']) as library, \
             VerifiedMatrixStore(maximum_matrix_bytes=store.maximum_matrix_bytes,
                                 maximum_source_bytes=store.limits['maximum_source_bytes']) as member_store:
            capabilities=library.describe()
            from .feature_catalog import load_feature_catalog_ref
            registered=load_feature_catalog_ref(spec['catalog_ref']).select(spec['feature_selection'])
            require([f['id'] for f in registered]==columns,'unregistered saved training projection selection')
            require(capabilities['sessions']==days and capabilities['sample_universe']==securities and
                    capabilities['columns']==columns and
                    [capabilities['definitions'][n]['schema'] for n in columns]==schema,
                    'training projection library binding mismatch')
            members=member_store.buffer(wire['source_member_buffer']).view('bool')
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

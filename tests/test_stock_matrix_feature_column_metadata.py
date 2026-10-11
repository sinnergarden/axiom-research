"""Synthetic physical-source closure and packed-result storage only."""
from copy import deepcopy
from pathlib import Path
import builtins
import struct
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import write_buffer, write_part, row_index, instant_us
from axiom_research.stock_matrix_feature_column_metadata import (
    SOURCE_BUFFERS, verify_feature_column_source, write_block, validate_metadata, admit_metadata, _core_ref)


def fixture(root, *, source_spec=None, qlib_view=None, output_session=None):
    days=['2023-01-02','2023-01-03']; securities=['A','B']
    spec={'read_sessions':days,'feature_sessions':days,'calendar':days,'universe':securities,'scope':{},
        'snapshot':'s_synthetic','pit_policy':'best_effort_vendor_v1',
        'cutoff_by_session':{d:d+'T12:30:00Z' for d in days},
        'catalog_ref':digest(['catalog']),'feature_selection':['f'],'ordered_features':['f']}
    if source_spec is not None:
        spec=deepcopy(source_spec); days=spec['read_sessions']; securities=spec['universe']
    view={'universe_name':'csi300'} if qlib_view is None else deepcopy(qlib_view)
    buffers={}
    for name,dtype in SOURCE_BUFFERS.items():
        shape=[len(days),len(securities)]+([] if name.startswith('member') else [6])
        count=len(days)*len(securities)*(1 if name.startswith('member') else 6)
        buffers[name]=write_buffer(root,[False if dtype=='bool_u8' else 0]*count,dtype=dtype,shape=shape)
    evidence=[]
    for domain in ['market_daily','adjustment_factors','universe_membership']:
        child=seal({'contract_version':'reader_column_evidence_v1','context':{'domain':domain},
            'field_meta':{},'fields':[],'sessions':days},'evidence_ref')
        evidence.append(write_part(root,child,'evidence_ref'))
    impl={'fixture':'no actual Feature execution'}
    manifest=seal({'contract_version':'research_feature_column_source_v1','spec':spec,'qlib_view':view,
        'buffers':buffers,'chunks':[{'sessions':days,'evidence':evidence}],
        'selection_invariance':'declared_vendor_release_visible_at_each_historical_row_cutoff',
        'implementation_sources':impl,'implementation_ref':digest(impl)},'source_ref')
    path=Path(root)/'source.json'; write_json(path,manifest)
    descriptor={'path':str(path),'file_digest':file_digest(path),'source_ref':manifest['source_ref']}
    schema=[{'name':spec['ordered_features'][0],'dtype':'float64','unit':'ratio','stage':'base','missing':'preserve'}]
    output_session=days[-1] if output_session is None else output_session
    history=days[:days.index(output_session)+1]
    selection={'source_ref':manifest['source_ref'],'session':output_session,'history_sessions':history,
        'anchor_session':output_session,'knowledge_cutoff':spec['cutoff_by_session'][output_session]}
    selection_ref=digest(selection)
    refs={'source_ref':manifest['source_ref'],'selection_ref':selection_ref}
    sources=[{'view_ref':manifest['source_ref']}]
    input_metadata={'input_refs':refs,'source_bindings':sources,'sessions':history,'security_ids':securities}
    input_buffers={'synthetic':{'shape':[len(history),len(securities)],'dtype':'<f8','sha256':'0'*64}}
    header={'contract_version':'feature_column_result_v1','plan_ref':digest(['synthetic plan']),
        'compiled_ref':digest(['synthetic compiled']),'implementation_ref':digest(['synthetic implementation']),
        'schema':schema,'input_refs':refs,'source_bindings':sources,
        'input_metadata':input_metadata,'input_buffers':input_buffers,
        'input_ref':_core_ref({'metadata':input_metadata,'buffers':input_buffers}),
        'sessions':[output_session],'security_ids':securities,'null_available_at_utc_us':-(2**63)}
    rows=[{'security_id':s,'session':output_session,'values':[value],'validity':[True],
        'availability':[output_session+'T12:00:00Z'],'reasons':[[]],'member':True,
        'knowledge_cutoff':selection['knowledge_cutoff'],'source_refs':[]}
        for s,value in zip(securities,[.1*(i+1) for i in range(len(securities))])]
    count=len(securities)
    payloads={'values':('<f8',struct.pack('<'+'d'*count,*[r['values'][0] for r in rows])),
        'validity':('|b1',b'\x01'*count),
        'available_at_utc_us':('<i8',struct.pack('<'+'q'*count,*[instant_us(r['availability'][0]) for r in rows])),
        'reason_codes':('<i4',b'\x00'*(count*4)),'source_codes':('<i4',b'\x00'*(count*4))}
    from hashlib import sha256
    header['output_buffers']={name:{'shape':[1,count,1],'dtype':dtype,'sha256':sha256(raw).hexdigest()}
        for name,(dtype,raw) in payloads.items()}
    header['result_ref']=_core_ref(header)
    for row in rows: row['source_refs']=[header['result_ref'],header['plan_ref']]
    proof=[{'selection':selection,'selection_ref':selection_ref,'core_header':header}]
    return descriptor,manifest,spec,view,schema,proof,rows


class ColumnMetadataTests(unittest.TestCase):
    def test_public_v4_feature_loader_and_sequential_table_projection(self):
        from test_stock_feature_inputs import SyntheticInputs, NoData
        from axiom_research import load_stock_feature_inputs
        from axiom_research.stock_compact_store import load_stock_feature_view
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=SyntheticInputs(root,feature_count=2); f.data=NoData()
            f.prepare(f.data,config={'read_sessions':f.calendar,'universe_id':f.universe_id},destination=root)
            physical=root/'physical'; physical.mkdir()
            descriptor,manifest,spec,view,schema,proof,rows=fixture(physical,source_spec=f.spec,
                qlib_view=f.view,output_session=f.days[0])
            spec=deepcopy(spec); spec['feature_sessions']=[f.days[0]]; spec['read_sessions']=f.calendar[:-1]
            spec['cutoff_by_session']={d:spec['cutoff_by_session'][d] for d in spec['read_sessions']}
            target=root/'saved'; target.mkdir(); options={'layout':'matrix_v3','row_block_sessions':1,
                'column_block':1,'maximum_resident_bytes':64*1024**2}
            index_rows=row_index(spec['feature_sessions'],spec['universe'])
            index_desc=write_part(target,index_rows,'row_index_ref')
            parts,selections=write_block(target,rows=rows,proof=proof,source_descriptor=descriptor,
                spec=spec,view=view,schema=schema,index_ref=index_rows['row_index_ref'],row_offset=0,
                options=options,universe_id=f.universe_id)
            implementation={'synthetic':digest(['no Feature computation'])}
            definition={'spec':spec,'storage_options':options,'implementation_sources':implementation,
                'implementation_ref':digest(implementation),'environment':{'synthetic':True},'column_source':descriptor}
            definition_ref=digest(definition)
            selected=write_part(target,seal({'contract_version':'stock_feature_source_selection_v3',
                'definition_ref':definition_ref,'feature_rows':selections},'source_selection_ref'),'source_selection_ref')
            qpath=root/'qlib'/'axiom-qlib.json'
            index={'contract_version':'stock_feature_inputs_v4','definition':definition,'definition_ref':definition_ref,
                'status':'COMPLETE','qlib_view':view,'qlib_manifest':{'path':str(qpath),'file_digest':file_digest(qpath),
                'view_id':f.manifest['view_id']},'schema':schema,'schema_digest':digest(schema),'row_index':index_desc,
                'source_selection':selected,'partitions':parts}
            index['feature_inputs_ref']=digest({k:index[k] for k in ('contract_version','definition_ref',
                'qlib_manifest','schema_digest','row_index','source_selection','partitions')})
            write_json(target/'index.json',seal(index,'content_digest'))
            with load_stock_feature_inputs(target) as saved:
                self.assertEqual(saved.identity,index['feature_inputs_ref'])
                self.assertEqual(saved.row_metadata([0,1,2]),rows)
                self.assertEqual(saved.to_dict()['definition']['spec']['feature_sessions'],[f.days[0]])
            with load_stock_feature_view(target,residency='sequential') as compact:
                self.assertEqual(compact.to_dict()['historical_feature_inputs_ref'],index['feature_inputs_ref'])
            bad=deepcopy(index); bad['definition']['spec']['cutoff_by_session'][f.days[0]]=f.days[0]+'T12:31:00Z'
            bad['definition_ref']=digest(bad['definition'])
            write_json(target/'index.json',seal(bad,'content_digest'))
            with self.assertRaisesRegex(ValueError,'clocks required|date/cutoff projection'):
                load_stock_feature_inputs(target)

    def test_store_shares_source_admission_and_rebuilds_daily_output_hashes(self):
        from axiom_research.stock_matrix_reader import VerifiedMatrixStore
        class Buffer:
            def __init__(self,desc):
                self.raw=Path(desc['path']).read_bytes(); self.width=8 if desc['dtype']!='bool_u8' else 1
            def __getitem__(self,index):
                rows,columns=index
                assert columns.start==0 and columns.stop==1
                value=Buffer.__new__(Buffer)
                value.raw=self.raw[rows.start*self.width:rows.stop*self.width]
                return value
            def tobytes(self,order):
                return self.raw
        with tempfile.TemporaryDirectory() as temp:
            descriptor,manifest,spec,view,schema,proof,rows=fixture(Path(temp))
            with VerifiedMatrixStore(maximum_matrix_bytes=64*1024**2) as store:
                store.read_json(descriptor,'source_ref')
                with patch('axiom_research.stock_parent_json.preflight_parent_file',
                           side_effect=AssertionError('column evidence used generic character scanner')):
                    verify_feature_column_source(descriptor,store=store)
                self.assertTrue(all(desc['path'] not in store.json and desc['path'] not in store._decoded
                    for chunk in manifest['chunks'] for desc in chunk['evidence']))
                hash_calls=store.metrics['file_hash_calls']; decodes=store.metrics['json_decode_calls']
                verify_feature_column_source(descriptor,store=store)
                self.assertEqual(store.metrics['file_hash_calls'],hash_calls)
                self.assertEqual(store.metrics['json_decode_calls'],decodes)
                parts,selections=write_block(temp,rows=rows,proof=proof,source_descriptor=descriptor,
                    spec=spec,view=view,schema=schema,index_ref=digest(['index']),row_offset=0,
                    options={'column_block':1,'maximum_resident_bytes':64*1024**2},
                    universe_id='csi300',source_store=store)
                metadata=_read(parts[0]['metadata']['path'])
                decoded=[(p,{key:Buffer(desc) for key,desc in p['buffers'].items()}) for p in parts]
                admitted,_,_=admit_metadata(store,metadata,rows,decoded,spec,view,universe_id='csi300')
                self.assertEqual(admitted,selections)

    def test_pure_hash_load_and_leaf_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            descriptor,manifest,*_=fixture(Path(temp)); original_import=builtins.__import__
            def blocked(name,*args,**kwargs):
                if name.startswith(('axiom_data','axiom_engine','numpy')):
                    raise AssertionError('unexpected business/numeric import '+name)
                return original_import(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=blocked):
                saved,reachable=verify_feature_column_source(descriptor)
                self.assertEqual(saved,manifest)
                self.assertIn(manifest['source_ref'],reachable)
            path=Path(manifest['buffers']['values']['path'])
            raw=path.read_bytes(); path.write_bytes(b'\x01'+raw[1:])
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_feature_column_source(descriptor)

    def test_packed_day_storage_identity_and_selection_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            descriptor,manifest,spec,view,schema,proof,rows=fixture(Path(temp))
            parts,selections=write_block(temp,rows=rows,proof=proof,source_descriptor=descriptor,
                spec=spec,view=view,schema=schema,index_ref=digest(['synthetic index']),row_offset=0,
                options={'column_block':1,'maximum_resident_bytes':64*1024**2},universe_id='csi300')
            metadata=_read(parts[0]['metadata']['path'])
            selected,reachable,parents=validate_metadata(metadata,rows,spec,view,manifest,
                metadata['day_outputs'],universe_id='csi300')
            self.assertEqual(selected,selections)
            self.assertEqual(selections[0]['feature_ref'],parents['2023-01-03']['feature_ref'])
            self.assertIn(proof[0]['selection_ref'],reachable)
            self.assertNotIn('core_frame_ref',metadata['input_evidence'][0])
            bad=deepcopy(metadata); p=bad['input_evidence'][0]
            p['core_header']['input_refs']=[descriptor['source_ref']]
            p.pop('day_evidence_ref'); p.update(seal(p,'day_evidence_ref'))
            bad.pop('metadata_ref'); bad.update(seal(bad,'metadata_ref'))
            with self.assertRaisesRegex(ValueError,'input selection refs'):
                validate_metadata(bad,rows,spec,view,manifest,bad['day_outputs'],universe_id='csi300')

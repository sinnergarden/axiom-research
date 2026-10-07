"""Small Label wire byte/identity/projection goldens; no Data or models."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from axiom_research.stock_artifacts import digest
from axiom_research.stock_label_wire import validate_label_wire_chunks
from axiom_research.stock_native_json import make_native_carrier
from axiom_research.stock_matrix_reader import VerifiedMatrixStore


WORKSPACE=4*1024**2


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def wire_fixture():
    symbols=['A','B']; sessions=['2026-01-02','2026-01-05','2026-01-06']
    wire={'context':{'query':{'symbols':symbols,'sessions':sessions,'fields':['open','close'],
        'cutoff_by_session':{d:'2026-01-06T12:00:00Z' for d in sessions}},
        'coverage':{'complete':True,'observations':[None,-0.0,'甲']}},
        'records':[],'field_meta':{f:{'dtype':'float64','unit':'CNY','by_key':[]} for f in ('open','close')}}
    for d in sessions:
        for s in symbols:
            wire['records'].append({'security_id':s,'session':d,'open':12.5,'close':13.0})
            for field in ('open','close'):
                leaf={'security_id':s,'session':d,'usable_from':'2026-01-06T11:00:00Z',
                      'missing_reason':None,'source_ref':'sha256:'+'a'*64}
                wire['field_meta'][field]['by_key'].append({'security_id':s,'session':d,'missing_reason':None,
                    'price_provenance':leaf,'factor_provenance':deepcopy(leaf),
                    'anchor_factor_provenance':{**leaf,'session':sessions[-1]}})
    return wire


def metadata(wire,role='training'):
    value={'contract_version':'stock_matrix_label_metadata_v1','role':role,
        'rows':[{'security_id':'A','start_session':'2026-01-02','end_session':'2026-01-05'}],
        'contents':{digest(wire):wire},'generation':'raw'}
    value['metadata_ref']=digest(value)
    return value


class LabelWireTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.wire=wire_fixture()

    def write(self,value,**kwargs):
        return make_native_carrier(self.root,value,'metadata_ref',maximum_source_bytes=1024**2,
            maximum_parent_bytes=2200,maximum_workspace_bytes=WORKSPACE,
            externalize_label_wire=True,**kwargs)

    def store(self,**overrides):
        store=VerifiedMatrixStore(maximum_source_bytes=1024**2,maximum_matrix_bytes=WORKSPACE,
            maximum_parent_bytes=2200,**overrides)
        self.addCleanup(lambda: None if store.closed else store.close())
        return store

    def test_components_and_context_exact_at_every_small_chunk_boundary(self):
        payload=canonical(self.wire)
        for n in (1,7,31,65536):
            proof=validate_label_wire_chunks((payload[i:i+n] for i in range(0,len(payload),n)),
                maximum_workspace_bytes=WORKSPACE)
            self.assertEqual(proof['digest'],digest(self.wire))
            self.assertEqual(proof['component_refs'],{k:digest(v) for k,v in self.wire.items()})
            self.assertEqual(proof['coverage_identity'],(True,digest(self.wire['context']['coverage'])))
            self.assertEqual(proof['context'],{k:v for k,v in self.wire['context'].items() if k!='coverage'})
            self.assertNotIn('endpoint_wire',proof)

    def test_endpoint_projection_never_decodes_the_whole_wire(self):
        payload=canonical(self.wire); wanted={('A','2026-01-05')}
        original=json.loads
        def guarded(value,*args,**kwargs):
            if isinstance(value,(bytes,bytearray)) and value.startswith(b'{"context":'):
                self.fail('whole selected-wire decode')
            return original(value,*args,**kwargs)
        with patch('axiom_research.stock_label_wire.json.loads',side_effect=guarded):
            proof=validate_label_wire_chunks((payload,),maximum_workspace_bytes=WORKSPACE,endpoint_keys=wanted)
        projected=proof['endpoint_wire']
        self.assertEqual(projected['records'],[r for r in self.wire['records']
            if (r['security_id'],r['session']) in wanted])
        for field in ('open','close'):
            self.assertEqual(projected['field_meta'][field],{**self.wire['field_meta'][field],
                'by_key':[r for r in self.wire['field_meta'][field]['by_key']
                          if (r['security_id'],r['session']) in wanted]})

    def test_v2_fits_parent_and_preserves_exact_original_native_identity(self):
        value=metadata(self.wire); original=deepcopy(value)
        with self.assertRaisesRegex(ValueError,'carrier parent byte budget exceeded'):
            make_native_carrier(self.root,value,'metadata_ref',maximum_source_bytes=1024**2,
                maximum_parent_bytes=2200,maximum_workspace_bytes=WORKSPACE)
        desc=self.write(value); carrier=json.loads(Path(desc['path']).read_bytes())
        self.assertEqual(carrier['contract_version'],'stock_native_json_carrier_v2')
        blob=carrier['wire_slots'][0]['bytes']
        self.assertEqual(Path(blob['path']).read_bytes(),canonical(self.wire))
        self.assertEqual(blob['buffer_digest'],digest(self.wire))
        store=self.store(); saved=store.read_json(desc,'metadata_ref')
        self.assertEqual(store.native_digest(saved,exclude_ref_key='metadata_ref'),original['metadata_ref'])
        self.assertEqual(store.native_digest(saved['contents'][digest(self.wire)]),digest(self.wire))
        self.assertEqual(value,original)
        self.assertIsNone(store._np)

    def test_normalized_metadata_reuses_original_wire_blob_without_encode(self):
        desc=self.write(metadata(self.wire)); store=self.store(); saved=store.read_json(desc,'metadata_ref')
        normalized={**saved,'generation':'normalized'}; normalized.pop('metadata_ref')
        normalized['metadata_ref']=store.native_digest(normalized)
        bindings=tuple(p for pairs in store._wire_bindings.values() for p in pairs); metrics={}
        second=self.write(normalized,wire_bindings=bindings,metrics=metrics)
        first_blob=json.loads(Path(desc['path']).read_bytes())['wire_slots'][0]['bytes']
        self.assertEqual(first_blob,json.loads(Path(second['path']).read_bytes())['wire_slots'][0]['bytes'])
        self.assertEqual(metrics.get('label_wire_encode_calls',0),0)
        store.read_json(second,'metadata_ref')
        self.assertEqual(store.metrics['label_wire_validation_calls'],1)

    def test_query_grid_mismatch_and_duplicate_rejected_from_actual_bytes(self):
        for mutate in (lambda w:w['records'].pop(),
                       lambda w:w['records'].__setitem__(1,deepcopy(w['records'][0])),
                       lambda w:w['field_meta']['open']['by_key'].pop()):
            wire=deepcopy(self.wire); mutate(wire)
            desc=self.write(metadata(wire))
            with self.assertRaisesRegex(ValueError,'selected Label.*(grid|key)'):
                self.store().read_json(desc,'metadata_ref')

    def test_slot_scope_duplicate_wrong_ref_and_placeholder_are_not_credentials(self):
        desc=self.write(metadata(self.wire)); original=json.loads(Path(desc['path']).read_bytes())
        mutations=(lambda c:c['wire_slots'].append(deepcopy(c['wire_slots'][0])),
                   lambda c:c['wire_slots'][0]['path'].__setitem__(0,'rows'),
                   lambda c:c['wire_slots'][0]['bytes'].__setitem__('buffer_digest','sha256:'+'b'*64),
                   lambda c:c['skeleton']['contents'][digest(self.wire)]['context'].__setitem__('fake',True))
        for i,mutate in enumerate(mutations):
            carrier=deepcopy(original); mutate(carrier); carrier.pop('carrier_ref'); carrier['carrier_ref']=digest(carrier)
            payload=canonical(carrier)+b'\n'; path=self.root/f'bad-{i}.json'; path.write_bytes(payload)
            bad={'path':str(path),'file_digest':'sha256:'+sha256(payload).hexdigest(),'metadata_ref':desc['metadata_ref']}
            with self.assertRaises(ValueError): self.store().read_json(bad,'metadata_ref')

    def test_real_wire_mutation_is_rejected_after_admission(self):
        desc=self.write(metadata(self.wire)); store=self.store(); store.read_json(desc,'metadata_ref')
        blob=json.loads(Path(desc['path']).read_bytes())['wire_slots'][0]['bytes']; path=Path(blob['path'])
        data=path.read_bytes(); path.write_bytes(data.replace(b'12.5',b'12.6'))
        with self.assertRaisesRegex(ValueError,'source changed'): store.read_json(desc,'metadata_ref')

    def test_admitted_placeholder_context_cannot_be_changed_into_a_clock_credential(self):
        desc=self.write(metadata(self.wire)); store=self.store(); saved=store.read_json(desc,'metadata_ref')
        child=saved['contents'][digest(self.wire)]
        self.assertTrue(store.context_equal(child['context'],self.wire['context']))
        child['context']['query']['cutoff_by_session']['2026-01-02']='2026-01-02T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'placeholder changed'):
            store.native_digest(saved,exclude_ref_key='metadata_ref')

    def test_public_evaluation_admission_retains_only_its_requested_endpoint_cells(self):
        desc=self.write(metadata(self.wire,role='evaluation')); store=self.store()
        child=store.read_json(desc,'metadata_ref')['contents'][digest(self.wire)]
        proof=store._label_wire_proof(child)
        wanted={('A','2026-01-02'),('A','2026-01-05')}
        self.assertEqual({(r['security_id'],r['session']) for r in proof['endpoint_wire']['records']},wanted)
        self.assertEqual(proof['component_refs']['field_meta'],digest(self.wire['field_meta']))

    def test_explicit_low_source_and_projection_budgets_reject(self):
        payload=canonical(self.wire)
        with self.assertRaisesRegex(ValueError,'workspace budget'):
            validate_label_wire_chunks((payload,),maximum_workspace_bytes=20000)
        desc=self.write(metadata(self.wire))
        store=VerifiedMatrixStore(maximum_source_bytes=2200,maximum_matrix_bytes=WORKSPACE,maximum_parent_bytes=2200)
        self.addCleanup(lambda: None if store.closed else store.close())
        with self.assertRaisesRegex(ValueError,'source byte budget'): store.read_json(desc,'metadata_ref')

    def test_publisher_counts_shared_wire_once_against_cumulative_source_budget(self):
        from axiom_research.stock_matrix_prepare import _Publisher
        stage=self.root/'stage'; stage.mkdir(); publisher=_Publisher(stage,self.root/'final',
            maximum_resident_bytes=WORKSPACE,maximum_source_bytes=1024**2,known_source_bytes=123)
        self.addCleanup(lambda: None if publisher.store.closed else publisher.store.close())
        first=publisher.part(metadata(self.wire),'metadata_ref')
        saved=publisher.read_part(first,'metadata_ref'); value={**saved,'generation':'normalized'}
        value=publisher.seal(value,'metadata_ref'); second=publisher.part(value,'metadata_ref')
        actual=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
        self.assertEqual(publisher.written_bytes,actual)
        self.assertEqual(len(list((stage/'buffers').glob('*.bin'))),1)
        publisher.store.maximum_source_bytes=publisher.known_source_bytes+actual
        with self.assertRaisesRegex(ValueError,'cumulative source byte budget'):
            publisher.buffer([1.0],dtype='float64_le',shape=[1])

    def test_staged_source_parent_and_matrix_limits_forward_exactly(self):
        from axiom_research.stock_matrix_reader import _validate_staged_matrix_batch
        stage=self.root/'complete'; stage.mkdir(); limits=[]
        def state(manifest,*,_store):
            limits.append((_store.maximum_source_bytes,_store.maximum_parent_bytes,_store.maximum_matrix_bytes))
            return SimpleNamespace(store=_store,close=_store.close)
        with patch('axiom_research.stock_matrix_reader.load_matrix_batch_state',side_effect=state):
            _validate_staged_matrix_batch({},stage=stage,target=self.root/'unpublished',
                maximum_source_bytes=123456,maximum_parent_bytes=98765,maximum_matrix_bytes=WORKSPACE)
        self.assertEqual(limits,[(123456,98765,WORKSPACE)])

    def test_physical_plan_changes_counts_without_changing_business_grid(self):
        from axiom_research.stock_matrix_prepare import _label_work_plan
        days=['2026-01-'+str(n).zfill(2) for n in range(2,12)]
        spec={'calendar':days,'feature_sessions':days,'universe':['A','B'],
              'read_sessions':days,'pit_policy':'operational_pit_v1'}
        fold={'fit_cutoff':days[5]+'T20:00:00Z','evaluation_cutoff':days[-1]+'T20:00:00Z'}
        with patch('axiom_research.stock_matrix_prepare.validate_spec',return_value=(days[:6],days[6:9])):
            plans=[_label_work_plan(spec,[fold],n) for n in (1,3,64)]
        self.assertEqual([p['raw_rows'] for p in plans],[18,18,18])
        self.assertEqual([p['raw_calls'] for p in plans],[9,3,2])
        self.assertEqual([p['data_calls'] for p in plans],[18,6,4])
        self.assertEqual([p['core_calls'] for p in plans],[1,1,1])


if __name__=='__main__': unittest.main()

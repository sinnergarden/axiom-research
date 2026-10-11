"""Publication races, bounded cold checkpoints and original file epochs."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import os
import sys
import tempfile
import unittest
import numpy as np

from axiom_research.stock_artifacts import digest, file_digest
from axiom_research.stock_compact_batch import TargetRows, read_target
from axiom_research.stock_compact_labels import (
    _publish, _raw, _column_raw_implementation, _column_normalization_implementation,
    _prepare_compact_incrementally, _IncrementalPreparation,
)
from axiom_research.stock_compact_store import OwnedStore
from axiom_research.stock_column_inputs import ColumnPriceDomain
from axiom_research.stock_matrix_storage import instant_us, write_buffer, write_part, _verify_written
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_target_spec import resolve_stock_label_spec
from test_stock_compact_v3 import CompactV3Tests
from test_stock_column_asof import ColumnSource, Query
from test_stock_target_config_blocks import label


def tiny_raw():
    calendar=['2024-01-'+str(i).zfill(2) for i in range(1,11)]
    days=calendar[:4];cutoff=calendar[-1]+'T20:30:00+08:00'
    definitions=dict(calendar=calendar,universe=['A'],sessions=days,cutoff=cutoff,
        horizon_sessions=5,snapshot='s_fixed')
    vectors={name:np.frombuffer(np.asarray(values,dtype=dtype).tobytes(),dtype=dtype)
        for name,values,dtype in (
            ('values',[.1]*4,'<f8'),('validity',[1]*4,'u1'),
            ('availability',[instant_us(cutoff)]*4,'<i8'),
            ('availability_validity',[1]*4,'u1'),('reason_codes',[0]*4,'<i4'),
            ('start_session',[1,2,3,4],'<i4'),('end_session',[5,6,7,8],'<i4'),
            ('source_codes',[0]*4,'<i4'))}
    return definitions,TargetRows(dict(contract_version='stock_compact_raw_v2',
        definition=definitions,row_count=4,reason_dictionary=[None],
        source_dictionary=[[digest('original Raw source')]]),vectors)


class SharedReviewFixTests(unittest.TestCase):
    def test_actual_publication_rejects_json_and_buffer_changes_after_writer_hash(self):
        original=OwnedStore.adopt_written
        for kind in ('json','buffer'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);definition,rows=tiny_raw();target=root/'target'
                def changed(store,descriptor,value,arrays,*,marks):
                    locator=descriptor['path'] if kind=='json' else value['buffers']['values']['path']
                    physical=store.resolve(locator)
                    physical.write_bytes(physical.read_bytes()+b' ')
                    return original(store,descriptor,value,arrays,marks=marks)
                with OwnedStore() as store,patch.object(OwnedStore,'adopt_written',changed):
                    with self.assertRaisesRegex(ValueError,'changed after byte verification'):
                        _publish(target,definition,rows,budgets=store.limits,store=store)
                    self.assertFalse(target.exists())
                    self.assertEqual((store.json,store.arrays,store.resident_bytes),({},{},0))

    def test_plain_json_handoff_uses_epoch_after_temporary_link_cleanup(self):
        with tempfile.TemporaryDirectory() as temp,OwnedStore() as store:
            marks={};value=seal({'contract_version':'small_proof_v1','value':1},'proof_ref')
            descriptor=write_part(temp,value,'proof_ref',_verified_marks=marks)
            store.adopt_written(descriptor,value,{},marks=marks)
            self.assertEqual(store.read_json(descriptor,key='proof_ref'),value)
            other=seal({'contract_version':'small_proof_v1','value':2},'proof_ref')
            changed=write_part(temp,other,'proof_ref',_verified_marks=marks)
            Path(changed['path']).write_bytes(Path(changed['path']).read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'changed after byte verification'):
                store.adopt_written(changed,other,{},marks=marks)

    def test_writer_detects_change_inside_final_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'output';path.write_bytes(b'original')
            expected=file_digest(path)
            def changed(locator):
                result=file_digest(locator);Path(locator).write_bytes(b'replaced');return result
            marks={}
            with patch('axiom_research.stock_matrix_storage.file_digest',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'changed during final verification'):
                    _verify_written(path,expected,marks)
            self.assertEqual(marks,{})

    def test_evicted_payload_rehydrates_original_epoch_with_one_initial_hash(self):
        with tempfile.TemporaryDirectory() as temp,OwnedStore() as store:
            descriptor=write_buffer(temp,[1.25],dtype='float64_le',shape=[1])
            self.assertEqual(store.buffer(descriptor).tolist(),[1.25])
            epoch=store.marks[descriptor['path']]
            store.release_payloads()
            self.assertEqual(store.resident_bytes,0)
            with patch('axiom_research.stock_compact_store.sha256',side_effect=AssertionError('duplicate full file hash')):
                self.assertEqual(store.buffer(descriptor).tolist(),[1.25])
            self.assertEqual(store.marks[descriptor['path']],epoch)
            self.assertEqual(store.metrics['file_hash_calls'],1)
            self.assertEqual(store.metrics['source_bytes'],8)
            self.assertEqual(store.metrics['file_read_calls'],2)
            self.assertEqual(store.metrics['rehydration_bytes'],8)
            store.release_payloads()
            os.utime(descriptor['path'],None)
            with self.assertRaisesRegex(ValueError,'source changed'):
                store.buffer(descriptor)

    def test_rehydration_rejects_change_during_read_even_inside_one_operation(self):
        with tempfile.TemporaryDirectory() as temp,OwnedStore() as store:
            descriptor=write_buffer(temp,[1.25],dtype='float64_le',shape=[1])
            store.buffer(descriptor);store.release_payloads()
            original_open=Path.open
            class ChangedRead:
                def __init__(self,stream):self.stream=stream
                def __enter__(self):self.stream.__enter__();return self
                def __exit__(self,*args):return self.stream.__exit__(*args)
                def fileno(self):return self.stream.fileno()
                def read(self,*args):
                    payload=self.stream.read(*args);os.utime(descriptor['path'],None);return payload
            def changed(path,*args,**kwargs):return ChangedRead(original_open(path,*args,**kwargs))
            with patch.object(Path,'open',changed),self.assertRaisesRegex(ValueError,'source changed'):
                with store.operation():
                    store.check_path(descriptor['path'])
                    store.buffer(descriptor)

    def test_columnar_raw_hit_fits_without_a_row_dictionary_reservation(self):
        with tempfile.TemporaryDirectory() as temp,OwnedStore() as store:
            cache=Path(temp);definition,rows=tiny_raw();cutoff=definition['cutoff'];days=definition['sessions']
            store.target_views={}
            spec={**definition,'feature_sessions':days,'target_spec':resolve_stock_label_spec(label(5))}
            source={'price_view_ref':digest('fixed selected domain')}
            expected={k:definition[k] for k in ('snapshot','calendar','universe','sessions','cutoff')}
            expected.update(price_view=source,formula='close(f+5) / open(f+1) - 1',
                price_basis='common_anchor_adjusted_v1',horizon_sessions=5,start_session_offset=1,end_session_offset=5,
                missing_policy='invalid_null_preserve_grid',implementation_ref=_column_raw_implementation(),
                label_definition_ref=spec['target_spec']['label_definition_ref'])
            descriptor=_publish(cache/'raw'/digest(expected)[7:],expected,rows,budgets=store.limits,store=store)
            _,admitted=read_target(store,descriptor,expected=expected)
            price=object.__new__(ColumnPriceDomain);price.spec=spec;price.source=source
            maximum=store.maximum_matrix_bytes;other_owner=maximum-store.resident_bytes-2048
            metrics={'_feature_store':SimpleNamespace(resident_bytes=0,metrics={'source_bytes':0}),
                '_store':store,'_caller_bytes':other_owner,'_caller_source_bytes':0,
                '_retained_raw_bytes':0,'_limits':store.limits,'maximum_working_bytes':0,'raw_cache_hits':0}
            saved,borrowed=_raw(spec,cutoff,days,cache,metrics,price)
            self.assertEqual(saved,descriptor);self.assertIs(borrowed,admitted)
            self.assertEqual(metrics['raw_cache_hits'],1)
            self.assertLessEqual(metrics['maximum_working_bytes'],maximum)

    def test_partial_column_checkpoint_drops_each_cold_raw_view_before_next(self):
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture,path=CompactV3Tests().fixture(root);source=ColumnSource(fixture.spec)
            options=dict(row_block_sessions=10,column_block=32,maximum_resident_bytes=64*1024**2,
                normalization_backend='core_cs_batch_v1')
            kwargs=dict(feature_inputs=path,fold_specs=fixture.folds(),destination=root/'prepared',
                label_spec=label(5),column_source=source,preparation_options=options)
            with patch.dict(sys.modules,{'axiom_data':module}):
                with _prepare_compact_incrementally(object(),**kwargs) as owner:
                    owner.next_fold();self.assertFalse((owner.target/'batch.json').exists())
                original=_IncrementalPreparation._release_payloads;counts=[]
                def observed(owner,**options):
                    if not options.get('keep_active',False):counts.append(len(owner.state.targets))
                    result=original(owner,**options)
                    if not options.get('keep_active',False):self.assertEqual(owner.state.targets,{})
                    return result
                with patch.object(_IncrementalPreparation,'_release_payloads',observed), \
                    patch.object(source,'select',side_effect=AssertionError('cold checkpoint selected Data')):
                    with _prepare_compact_incrementally(object(),**kwargs) as owner:
                        self.assertEqual(owner.state.targets,{})
                self.assertGreater(len(counts),2);self.assertEqual(max(counts),1)

    def test_raw_and_normalized_identities_bind_actual_vector_dependencies(self):
        import axiom_research.stock_compact_batch as targets
        import axiom_research.stock_compact_store as features
        def changed_columns(*args,**kwargs):raise AssertionError('identity-only mutation')
        def changed_eligibility(*args,**kwargs):raise AssertionError('identity-only mutation')
        raw=_column_raw_implementation();normalized=_column_normalization_implementation()
        with patch.object(ColumnPriceDomain,'columns',changed_columns):
            self.assertNotEqual(_column_raw_implementation(),raw)
        with patch.object(targets,'target_eligibility_reasons',changed_eligibility):
            self.assertNotEqual(_column_normalization_implementation(),normalized)
        with patch.object(features,'feature_eligibility_columns',changed_eligibility):
            self.assertNotEqual(_column_normalization_implementation(),normalized)


if __name__=='__main__':unittest.main()

"""Tiny sequential owner windows; no real data or account execution."""
from copy import deepcopy
from pathlib import Path
from contextlib import ExitStack
import gc
import tempfile
import unittest
import weakref
from unittest.mock import patch

import test_stock_compact_v3 as fixtures
from test_stock_folds import backend
from axiom_research import (load_stock_feature_view, load_stock_ml_batch_inputs,
    build_stock_ml_fold_from_saved_inputs)
from axiom_research.stock_artifacts import _read, digest, file_digest, write_json
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_store import OwnedStore, _view_data
from axiom_research.stock_fold_inputs import seal, validate_spec
from axiom_research.stock_matrix_storage import instant_us, write_buffer, write_part


class SequentialWindowTests(unittest.TestCase):
    def fixture(self, root, *, folds=2):
        """One day per block: shared values, independently timed metadata."""
        owner=fixtures.CompactV3Tests(); f,path=owner.fixture(root)
        index=_read(path/'index.json'); parts=[]; width=len(f.universe)
        for offset, day in enumerate(f.days):
            rows,proof=f.day(day)
            parents={day:{'feature_ref':digest(proof),'qlib_view_ref':digest(['saved-qlib',day])}}
            metadata=write_part(path,seal({'rows':rows,'row_references':parents},'metadata_ref'),'metadata_ref')
            shape=[width,len(f.columns)]
            buffers={name:write_buffer(path,values,dtype=dtype,shape=shape) for name,dtype,values in (
                ('values','float64_le',[v if flag else 0.0 for row in rows for v,flag in zip(row['values'],row['validity'])]),
                ('value_validity','bool_u8',[flag for row in rows for flag in row['validity']]),
                ('available_at_utc_us','int64_le',[instant_us(at) if at is not None else 0 for row in rows for at in row['availability']]),
                ('available_at_validity','bool_u8',[at is not None for row in rows for at in row['availability']]))}
            parts.append(seal({'table':'features','row_index_ref':index['row_index']['row_index_ref'],
                'schema_digest':digest(index['schema']),'row_offset':offset*width,'row_count':width,
                'columns':f.columns,'buffers':buffers,'metadata':metadata},'partition_ref'))
        index['partitions']=parts; write_json(path/'index.json',index)
        with load_stock_feature_view(path) as view:
            manifest,data=owner.prepare(f,view,root,folds=f.folds()[:folds])
        return f,path,manifest,data

    def build(self, f, fold, batch, destination, *, fit=backend, metrics=None):
        with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
             patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
             patch('axiom_research.stock_ml._environment',return_value=f.environment), \
             patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=fit):
            return build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                destination=destination,batch=batch,metrics=metrics)

    def reseal_controls(self, manifest, view):
        """Create deliberately changed, consistently hashed control documents."""
        manifest=deepcopy(manifest); view=deepcopy(view)
        view=seal({k:v for k,v in view.items() if k!='prepared_view_ref'},'prepared_view_ref')
        path=Path(manifest['prepared_view']['path']); write_json(path,view)
        descriptor={'path':str(path),'file_digest':file_digest(path),'prepared_view_ref':view['prepared_view_ref']}
        manifest['prepared_view']=descriptor
        manifest['definition']['fold_specs']=[fold['fold_spec'] for fold in manifest['folds']]
        manifest['definition_ref']=digest(manifest['definition'])
        for fold in manifest['folds']:
            inputs=fold['input_manifest']; inputs['prepared_view']=deepcopy(descriptor)
            inputs['fold_spec_ref']=digest(fold['fold_spec'])
            fold['input_manifest']=seal({k:v for k,v in inputs.items() if k!='input_ref'},'input_ref')
        manifest={k:v for k,v in manifest.items() if k not in ('batch_ref','content_digest')}
        manifest['batch_ref']=digest(manifest); manifest['content_digest']=digest(manifest)
        return manifest

    def test_initial_controls_do_not_admit_feature_or_target_buffers(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.fixture(Path(temp))
            frozen=digest(manifest)
            with patch.object(OwnedStore,'buffer',side_effect=AssertionError('initial eager leaf admission')):
                batch=load_stock_ml_batch_inputs(manifest,residency='sequential')
            with batch:
                state=_data(batch)['matrix_state']; fd=_view_data(state.feature)
                self.assertEqual(state.targets,{})
                self.assertEqual((state.store.arrays,fd['store'].arrays),({},{}))
                self.assertTrue(all(block['parts'] is None for block in fd['blocks']))
                self.assertEqual(batch.identity,manifest['batch_ref'])
                self.assertEqual(digest(batch.to_dict()),frozen)
                self.assertEqual(state.residency,'sequential')
                self.assertEqual(fd['residency'],'sequential')

    def test_fold_matrices_match_eager_bytes_and_independent_order(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.fixture(Path(temp))
            with load_stock_ml_batch_inputs(manifest) as eager, \
                 load_stock_ml_batch_inputs(manifest,residency='sequential') as sequential:
                for fold in manifest['folds']:
                    with eager._matrix_project(fold['input_manifest'],fold['fold_spec']) as expected, \
                         sequential._matrix_project(fold['input_manifest'],fold['fold_spec']) as actual:
                        self.assertEqual(actual.training_keys,expected.training_keys)
                        self.assertEqual(actual.candidate_keys,expected.candidate_keys)
                        self.assertEqual(actual.features,expected.features)
                        self.assertEqual(actual.labels,expected.labels)
                        self.assertEqual(actual.evaluation,expected.evaluation)
                        for name in ('X','y','P'):
                            a,b=getattr(actual,name),getattr(expected,name)
                            self.assertEqual((a.shape,a.dtype,a.tobytes()),(b.shape,b.dtype,b.tobytes()))
                            self.assertFalse(a.flags.writeable)
                        training,inference=validate_spec(fold['fold_spec'],f.calendar)
                        mature=[day for day in training if f.calendar[f.calendar.index(day)+5]<=fold['fold_spec']['fit_session']]
                        keys=[[security,day] for day in mature for security in ('A','B')]
                        self.assertEqual(actual.training_keys,keys)
                        self.assertEqual(actual.candidate_keys,[[security,day] for day in inference for security in ('A','B')])
                        self.assertEqual(actual.X.tobytes(),np.asarray([[float(i+1) for i in range(6)]]*len(keys),dtype='<f8').tobytes())
                        self.assertEqual(actual.P.tobytes(),np.asarray([[float(i+1) for i in range(6)]]*(len(inference)*2),dtype='<f8').tobytes())
                        # Independent two-member population normalization sign.
                        np.testing.assert_allclose(actual.y,np.tile([-1.0,1.0],len(mature)),rtol=0,atol=2e-14)

    def test_switch_releases_views_and_actual_charges_reuses_shared_leaf(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.fixture(Path(temp))
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; fd=_view_data(state.feature); fstore=fd['store']
                first,second=manifest['folds']; released=[]; real_release=OwnedStore.release
                def release(store,paths):
                    before=store.resident_bytes; buffers={p:len(value) for p,value in store.arrays.items()}
                    result=real_release(store,paths)
                    removed=set(buffers)-set(store.arrays)
                    self.assertEqual(result['buffer_bytes'],sum(buffers[p] for p in removed))
                    self.assertEqual(before-store.resident_bytes,result['buffer_bytes']+result['json_bytes'])
                    released.append((store,result)); return result
                with batch._matrix_project(first['input_manifest'],first['fold_spec']) as projection:
                    matrix_refs=[weakref.ref(getattr(projection,name)) for name in ('X','y','P')]
                    block=fd['blocks'][0]
                    feature_refs=[weakref.ref(array) for _,arrays in block['parts'] for array in arrays.values()]
                    target_refs=[weakref.ref(rows) for _,rows in state.targets.values()]
                    target_array_refs=[weakref.ref(array) for _,rows in state.targets.values() for array in rows.arrays.values()]
                    shared_path=block['descriptors'][0]['buffers']['values']['path']
                    shared_identity=id(fstore.arrays[shared_path]); first_hashes=fstore.metrics['file_hash_calls']
                    excluded_path=block['descriptors'][0]['buffers']['available_at_utc_us']['path']
                    self.assertIn(excluded_path,fstore.arrays)
                self.assertTrue(all(ref() is None for ref in matrix_refs))
                self.assertEqual((state.store.borrowers,state.active,state.store.lease_bytes),(0,0,0))
                with patch.object(OwnedStore,'release',new=release), \
                     batch._matrix_project(second['input_manifest'],second['fold_spec']):
                    self.assertNotIn(excluded_path,fstore.arrays)
                    self.assertTrue(all(ref() is None for ref in feature_refs+target_refs+target_array_refs))
                    self.assertEqual(id(fstore.arrays[shared_path]),shared_identity)
                    self.assertEqual(fstore.metrics['file_hash_calls']-first_hashes,6,
                        'only three new day metadata and availability leaves should hash')
                self.assertTrue(any(result['buffer_bytes']>0 for _,result in released))
                self.assertGreater(fstore.metrics['released_buffer_bytes'],0)
                self.assertGreater(state.store.metrics['released_buffer_bytes'],0)
                state.verify_all()
                self.assertEqual((state.targets,state.store.arrays,fstore.arrays),({},{},{}))
                self.assertEqual(state.store.metrics['verified_fold_count'],2)

    def test_active_projection_prevents_window_switch_and_owner_close(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.fixture(Path(temp))
            batch=load_stock_ml_batch_inputs(manifest,residency='sequential')
            state=_data(batch)['matrix_state']; first,second=manifest['folds']
            try:
                with batch._matrix_project(first['input_manifest'],first['fold_spec']):
                    for operation in (batch.close,state._release_window,
                            lambda:batch._matrix_project(second['input_manifest'],second['fold_spec']),
                            state.feature.close):
                        with self.assertRaisesRegex(ValueError,'borrow'):
                            operation()
                with batch._matrix_project(second['input_manifest'],second['fold_spec']): pass
            finally: batch.close()

    def test_released_buffer_readmission_hashes_again_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); descriptor=write_buffer(root,[1.25,-3.5],dtype='float64_le',shape=[2])
            with OwnedStore() as store:
                array=store.buffer(descriptor); reference=weakref.ref(array); original=array.tobytes()
                del array
                release=store.release([descriptor['path']])
                self.assertIsNone(reference())
                self.assertEqual(release,{'buffer_bytes':16,'json_bytes':0})
                self.assertEqual(store.resident_bytes,0)
                self.assertNotIn(descriptor['path'],store.hashes)
                self.assertIn(descriptor['path'],store.marks)
                count=store.metrics['file_hash_calls']; array=store.buffer(descriptor)
                self.assertEqual(store.metrics['file_hash_calls'],count+1)
                self.assertEqual(array.tobytes(),original); del array
                store.release_payloads()
                leaf=Path(descriptor['path']); payload=leaf.read_bytes()
                leaf.write_bytes(bytes([payload[0]^1])+payload[1:])
                with self.assertRaisesRegex(ValueError,'changed|digest mismatch'):
                    store.buffer(descriptor)
                self.assertEqual((store.arrays,store.resident_bytes),({},0))

    def test_lazy_target_tampering_rejects_before_native_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.fixture(Path(temp),folds=1)
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; record=state.view['fold_targets'][0]
                target=_read(record['normalized']['path']); leaf=Path(target['buffers']['values']['path'])
                payload=leaf.read_bytes(); leaf.write_bytes(bytes([payload[0]^1])+payload[1:])
                with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('fit reached')), \
                     self.assertRaisesRegex(ValueError,'digest mismatch'):
                    self.build(f,manifest['folds'][0],batch,Path(temp)/'folds',fit=AssertionError('fit reached'))
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual((state.targets,state.store.arrays,_view_data(state.feature)['store'].arrays),({},{},{}))
                self.assertFalse((Path(temp)/'folds').exists())

    def test_control_validation_keeps_selector_clock_and_overlap_guards(self):
        for invalid in ('selector','clock','overlap'):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as temp:
                f,path,manifest,_=self.fixture(Path(temp)); view=_read(manifest['prepared_view']['path'])
                if invalid=='selector':
                    manifest['folds'][0]['input_manifest']['selectors']['training']=[]
                elif invalid=='clock':
                    manifest['folds'][0]['fold_spec']['fit_cutoff']=manifest['folds'][0]['fold_spec']['fit_session']+'T20:31:00+08:00'
                    view['fold_targets'][0]['fold_spec']=deepcopy(manifest['folds'][0]['fold_spec'])
                else:
                    manifest['folds'][1]=deepcopy(manifest['folds'][0])
                    view['fold_targets'][1]=deepcopy(view['fold_targets'][0])
                changed=self.reseal_controls(manifest,view)
                with patch.object(OwnedStore,'buffer',side_effect=AssertionError('invalid controls admitted leaves')), \
                     self.assertRaisesRegex(ValueError,'selector|clock|chronology'):
                    load_stock_ml_batch_inputs(changed,residency='sequential')

    def test_exact_saved_hit_uses_evaluation_projection_without_training_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path,manifest,_=self.fixture(root,folds=1); fold=manifest['folds'][0]
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                original=self.build(f,fold,batch,root/'folds')
                state=_data(batch)['matrix_state']; state._release_window(); stats={}
                before=state.store.metrics.get('matrix_allocated_bytes',0)
                with patch('axiom_research.stock_compact_batch.training_matrix',side_effect=AssertionError('HIT allocated X')):
                    saved=self.build(f,fold,batch,root/'folds',fit=AssertionError('HIT trained'),metrics=stats)
                self.assertTrue(saved.reused); self.assertEqual(saved.identity,original.identity)
                self.assertEqual((stats['train_calls'],stats['predict_calls'],stats['core_calls']),(0,0,0))
                self.assertEqual(state.store.metrics.get('matrix_allocated_bytes',0),before)
                self.assertGreater(state.store.metrics.get('evaluation_projection_calls',0),0)
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))

    def test_fit_and_publication_failures_release_projection_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path,manifest,_=self.fixture(root,folds=1); fold=manifest['folds'][0]
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']
                for failure in ('fit','publication'):
                    with self.subTest(failure=failure):
                        refs=[]
                        def fit(X,y,P,**kwargs):
                            refs.extend(weakref.ref(a) for a in (X,y,P))
                            if failure=='fit': raise RuntimeError('explicit fit failure')
                            return backend(X,y,P,**kwargs)
                        destination=root/('folds-'+failure)
                        with ExitStack() as stack:
                            if failure=='publication': stack.enter_context(patch.object(Path,'rename',side_effect=RuntimeError('explicit publication failure')))
                            with self.assertRaisesRegex(RuntimeError,'explicit '+failure+' failure'):
                                self.build(f,fold,batch,destination,fit=fit)
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in refs))
                        self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                        self.assertFalse(list(destination.glob('*/fold.json')))
                        state._release_window()
                        self.assertEqual((state.targets,state.store.arrays,_view_data(state.feature)['store'].arrays),({},{},{}))


if __name__=='__main__': unittest.main()

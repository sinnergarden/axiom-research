"""Bounded real owner/validator, artificial prices and the original fake backend."""
from pathlib import Path
from unittest.mock import patch
import gc
import tempfile
import unittest
import weakref

from axiom_research import load_stock_ml_batch_inputs,save_stock_signal_evaluation_inputs
from axiom_research.stock_artifacts import _read,file_digest
from axiom_research.stock_batch import _data,StockMLBatchInputs
from axiom_research.stock_signal_evaluation_build import _freeze_build_oos_inputs,_BuildOOSWriter
from axiom_research.stock_signal_evaluation_projection import _load_inputs
from axiom_research.stock_compact_store import OwnedStore
import test_stock_compact_v4 as compact_fixtures
import test_stock_sequential_windows as window_fixtures


class BuildOOSLifecycleTests(unittest.TestCase):
    def fixture(self,root):
        class Data(compact_fixtures.PublicDataFixture):
            def adjust(self,*args,**kwargs):
                value=super().adjust(*args,**kwargs)
                # The old preparation-only fixture omitted this real Data
                # derivation field; source admission must retain it exactly.
                value.wire['context']['derivation']['factor_domain']='adjustment_factors'
                return value
        with patch.object(compact_fixtures,'PublicDataFixture',Data):
            f,path,manifest,data=compact_fixtures.CompactV4Tests().prepare(root)
        days=sorted({d for item in manifest['folds'] for d in item['fold_spec']['inference_cutoff_by_session']})
        scope={'calendar':f.calendar,'sessions':days,'universe':f.universe,
               'evaluation_cutoff':manifest['folds'][0]['fold_spec']['evaluation_cutoff']}
        return f,manifest,scope

    def descriptor(self,run):
        p=run.path/'predictions.json'
        return {'path':str(p),'file_digest':file_digest(p),'signal_run_ref':run.predictions()['signal_run_ref']}

    def test_cold_borrows_live_projection_and_freezes_exact_default_rows_then_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root)
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; fixed=state._fixed_shared_bytes
                consume=_BuildOOSWriter._consume; borrowed=[]; reads=[]; read=OwnedStore.read
                def checked(writer,descriptor,lease):
                    projection=lease._projection
                    self.assertFalse(lease._owns_projection)
                    self.assertFalse(projection._closed)
                    self.assertEqual((projection.X,projection.y,projection.P),(None,None,None))
                    self.assertEqual(state.active,1)
                    consume(writer,descriptor,lease)
                    self.assertFalse(projection._closed)
                    borrowed.append(weakref.ref(projection))
                def counted(store,descriptor,**kwargs):
                    if '.matrix-fold-' in descriptor['path']: reads.append(descriptor['path'])
                    return read(store,descriptor,**kwargs)
                runs=[]; metrics=[]
                with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'stream',signal_name='model') as writer, \
                     patch.object(_BuildOOSWriter,'_consume',new=checked), \
                     patch.object(StockMLBatchInputs,'_project_evaluation',side_effect=AssertionError('second OOS projection')), \
                     patch.object(OwnedStore,'read',new=counted):
                    for item in manifest['folds']:
                        m={};runs.append(window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds',metrics=m));metrics.append(m)
                        self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                        self.assertFalse(hasattr(writer,'projected'))
                        self.assertTrue(all(ref() is None for ref in borrowed))
                    with patch('axiom_research.stock_signal_evaluation_projection._select_inputs',side_effect=AssertionError('full OOS allocation')):
                        streamed=writer.finish()
                    self.assertEqual(writer.metrics['build_projection_borrows'],2)
                    self.assertEqual(writer.metrics['maximum_fold_rows'],3*len(scope['universe']))
                    print('BUILD_OOS_LIFETIME '+str(writer.metrics))
                self.assertEqual(state._fixed_shared_bytes,fixed)
                self.assertEqual(len(reads),18)
                self.assertTrue(all(m['released_matrix_lease_bytes']>m['matrix_bytes'] for m in metrics))
                self.assertEqual(state.store.metrics['training_projection_calls'],2)
                self.assertNotIn('evaluation_projection_calls',state.store.metrics)
                descriptors=[self.descriptor(run) for run in runs]
                default=save_stock_signal_evaluation_inputs({'model':descriptors},scope=scope,
                    destination=root/'default',batch=batch)
                self.assertEqual(streamed.artifact_id,default.artifact_id)
                self.assertEqual(_read(streamed.uri),_read(default.uri))
                for day in scope['sessions']:
                    self.assertEqual((Path(streamed.uri).parent/(day+'.json')).read_bytes(),
                                     (Path(default.uri).parent/(day+'.json')).read_bytes())
                _,_,a,aa=_load_inputs(streamed,scope,include_admission=True)
                _,_,b,bb=_load_inputs(default,scope,include_admission=True)
                self.assertEqual(a,b);self.assertEqual(aa,bb)
                before=state.store.metrics['training_projection_calls']
                with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'hit',signal_name='model') as writer:
                    for item in manifest['folds']:
                        m={}
                        def forbidden(*a,**k):raise AssertionError('HIT fit/predict')
                        window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds',fit=forbidden,metrics=m)
                        self.assertTrue(m['cache_hit']);self.assertEqual(m['train_calls'],0)
                    hit=writer.finish();self.assertEqual(writer.metrics['saved_oos_admissions'],2)
                self.assertEqual(hit.artifact_id,streamed.artifact_id)
                self.assertEqual(state.store.metrics['training_projection_calls'],before)
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state._fixed_shared_bytes,fixed)

    def test_writer_lifetime_guards_owner_close_without_blocking_fold_activation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root)
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']
                with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'guard',signal_name='model') as writer:
                    with self.assertRaisesRegex(ValueError,'writer still borrowed'):batch.close()
                    with patch('axiom_research.stock_signal_evaluation_build.os.getpid',return_value=-1):
                        with self.assertRaisesRegex(ValueError,'another process'):writer.finish()
                    with self.assertRaisesRegex(ValueError,'one build OOS writer'):
                        with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'double',signal_name='model'):pass
                    window_fixtures.SequentialWindowTests().build(f,manifest['folds'][0],batch,root/'folds')
                    self.assertEqual(state.active,0)
                self.assertEqual(state._build_oos_borrowers,0)

    def test_concurrent_winner_is_freshly_admitted_with_same_build_projection(self):
        from axiom_research import stock_signal_evaluation_lease as lease_owner
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root);item=manifest['folds'][0]
            local={**scope,'sessions':sorted(item['fold_spec']['inference_cutoff_by_session'])}
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                saved=window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds')
                exists=Path.exists;calls=[0]
                def unseen(path):
                    if path==saved.path and calls[0]<2:calls[0]+=1;return False
                    return exists(path)
                with _freeze_build_oos_inputs(batch,scope=local,destination=root/'winner',signal_name='model') as writer, \
                     patch.object(Path,'exists',new=unseen), \
                     patch.object(lease_owner,'_admit_build_fold',wraps=lease_owner._admit_build_fold) as admissions:
                    result=window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds')
                    self.assertEqual(result.identity,saved.identity)
                    self.assertEqual(admissions.call_count,2)
                    self.assertIs(admissions.call_args_list[0].kwargs['projection'],
                                  admissions.call_args_list[1].kwargs['projection'])
                    self.assertEqual(writer.metrics['folds'],1)
                    writer.finish()
                state=_data(batch)['matrix_state']
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))

    def test_hit_consumer_failure_releases_saved_admission_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root);item=manifest['folds'][0]
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds')
                state=_data(batch)['matrix_state'];fixed=state._fixed_shared_bytes
                with self.assertRaisesRegex(RuntimeError,'HIT OOS failure'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'hit-error',signal_name='model'), \
                         patch('axiom_research.stock_signal_evaluation_compact._admit_compact',side_effect=RuntimeError('HIT OOS failure')):
                        window_fixtures.SequentialWindowTests().build(f,item,batch,root/'folds')
                self.assertEqual(state.targets,{})
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state._fixed_shared_bytes,fixed)

    def test_early_matrix_credit_survives_gc_without_double_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            f,manifest,scope=self.fixture(Path(temporary))
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; item=manifest['folds'][0]
                projection=batch._matrix_project(item['input_manifest'],item['fold_spec'])
                refs=[weakref.ref(getattr(projection,k)) for k in ('X','y','P')]
                original=state.store.lease_bytes; released=projection._release_matrices()
                self.assertGreater(released,0);self.assertEqual(state.store.lease_bytes,original-released)
                self.assertEqual(projection._release_matrices(),0)
                self.assertTrue(all(ref() is None for ref in refs))
                self.assertEqual(state.active,1)
                with self.assertRaisesRegex(ValueError,'borrow'):batch.close()
                del projection;gc.collect()
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))

    def test_incomplete_and_consumer_failure_never_publish_and_restore_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root)
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state'];fixed=state._fixed_shared_bytes
                with self.assertRaisesRegex(ValueError,'complete build OOS'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'incomplete',signal_name='model') as writer:
                        window_fixtures.SequentialWindowTests().build(f,manifest['folds'][0],batch,root/'folds')
                        writer.finish()
                self.assertEqual(list((root/'incomplete').iterdir()),[])
                self.assertEqual(state._fixed_shared_bytes,fixed)
                with self.assertRaisesRegex(RuntimeError,'OOS consumer failure'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'failure',signal_name='model'), \
                         patch('axiom_research.stock_signal_evaluation_compact._admit_compact',side_effect=RuntimeError('OOS consumer failure')):
                        window_fixtures.SequentialWindowTests().build(f,manifest['folds'][1],batch,root/'folds')
                self.assertEqual(list((root/'failure').iterdir()),[])
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state._fixed_shared_bytes,fixed)

    def test_prior_output_change_and_actual_child_budget_reject_before_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root)
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state'];fixed=state._fixed_shared_bytes
                with self.assertRaisesRegex(ValueError,'frozen evaluation file changed'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'changed',signal_name='model') as writer:
                        run=window_fixtures.SequentialWindowTests().build(f,manifest['folds'][0],batch,root/'folds')
                        p=run.path/'booster.txt';p.write_bytes(p.read_bytes())
                        writer.finish()
                self.assertEqual(list((root/'changed').iterdir()),[])
                consume=_BuildOOSWriter._consume
                def budget(writer,descriptor,lease):
                    previous=state.store.limits['maximum_matrix_bytes']
                    state.store.limits['maximum_matrix_bytes']=(state.store.shared_bytes+
                        state.store.resident_bytes+state.store.lease_bytes+1)
                    try:consume(writer,descriptor,lease)
                    finally:state.store.limits['maximum_matrix_bytes']=previous
                with self.assertRaisesRegex(ValueError,'resident byte budget'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'budget',signal_name='model'), \
                         patch.object(_BuildOOSWriter,'_consume',new=budget):
                        window_fixtures.SequentialWindowTests().build(f,manifest['folds'][1],batch,root/'folds')
                self.assertEqual(list((root/'budget').iterdir()),[])
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state._fixed_shared_bytes,fixed)

    def test_changed_output_fingerprint_during_directory_rename_is_not_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,manifest,scope=self.fixture(root);rename=Path.rename
            def changed(path,target):
                result=rename(path,target)
                if path.name=='complete' and '.matrix-fold-' in str(path):
                    p=Path(target)/'booster.txt';p.write_bytes(p.read_bytes())
                return result
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state'];fixed=state._fixed_shared_bytes
                with self.assertRaisesRegex(ValueError,'build output changed'):
                    with _freeze_build_oos_inputs(batch,scope=scope,destination=root/'rename',signal_name='model'), \
                         patch.object(Path,'rename',new=changed):
                        window_fixtures.SequentialWindowTests().build(f,manifest['folds'][0],batch,root/'folds')
                self.assertEqual(list((root/'rename').iterdir()),[])
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state._fixed_shared_bytes,fixed)



if __name__=='__main__':unittest.main()

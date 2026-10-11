"""Small saved v4 owner admission; no runtime work during lease consumption.

Fixture preparation uses the existing artificial public Data/Core oracle and
fake model backend. Every tested lease explicitly forbids those execution paths.
"""
from pathlib import Path
import builtins
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import load_stock_ml_batch_inputs,load_stock_ml_fold
from axiom_research.stock_matrix_folds import admit_stock_signal_evaluation_fold
from axiom_research.stock_artifacts import _read,file_digest
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_store import OwnedStore,_view_data
from axiom_research.stock_fold_inputs import file_fingerprint
from axiom_research.stock_signal_evaluation_inputs import _grid
import test_stock_compact_v4 as compact_fixtures
import test_stock_sequential_windows as window_fixtures


class FoldLeaseTests(unittest.TestCase):
    def fixture(self,root):
        f,path,manifest,_=compact_fixtures.CompactV4Tests().prepare(root)
        with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
            runs=[window_fixtures.SequentialWindowTests().build(f,fold,batch,root/'folds') for fold in manifest['folds']]
        return manifest,runs

    def descriptor(self,run):
        return {'path':str(run.path/'predictions.json'),'file_digest':file_digest(run.path/'predictions.json'),
                'signal_run_ref':_read(run.path/'predictions.json')['signal_run_ref']}

    def forbidden(self):
        from contextlib import ExitStack
        stack=ExitStack(); real=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.startswith(('axiom_data','axiom_engine','qlib','lightgbm','pandas','sklearn')):
                raise AssertionError('readonly lease imported '+name)
            return real(name,*args,**kwargs)
        stack.enter_context(patch('builtins.__import__',side_effect=imports))
        stack.enter_context(patch('axiom_research.stock_compact_batch.training_matrix',side_effect=AssertionError('training matrix')))
        stack.enter_context(patch('axiom_research.stock_compact_batch.CompactState.verify_all',side_effect=AssertionError('all-fold scan')))
        stack.enter_context(patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('fit/predict')))
        return stack

    def test_full_output_admission_once_and_readonly_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest,runs=self.fixture(Path(temporary)); descriptor=self.descriptor(runs[0]); hashes=[]
            original=OwnedStore.read
            def read(store,desc,**options):
                if Path(desc['path']).parent==runs[0].path: hashes.append(Path(desc['path']).name)
                return original(store,desc,**options)
            with self.forbidden(),load_stock_ml_batch_inputs(manifest,residency='sequential') as batch,patch.object(OwnedStore,'read',new=read):
                state=_data(batch)['matrix_state']; fixed=(state._fixed_shared_bytes,state._fixed_shared_source_bytes)
                with admit_stock_signal_evaluation_fold(descriptor,batch=batch) as lease:
                    self.assertEqual(set(lease.documents),{'manifest.json','fold.json','model.json','feature-slice.json','predictions.json'})
                    for name,value in lease.documents.items(): self.assertEqual(value,_read(runs[0].path/name))
                    self.assertFalse(any(hasattr(lease,key) for key in ('X','y','P','booster','training_keys')))
                    records=dict(lease.source_records)
                    fingerprints=dict(lease.source_fingerprints)
                    self.assertEqual(set(lease.source_fingerprints),set(records))
                    self.assertTrue(all(file_digest(path)==ref for path,ref in records.items()))
                    self.assertGreater(state.store.lease_bytes,0)
                    with self.assertRaisesRegex(ValueError,'borrow'): batch.close()
                    source=lease.evaluation_targets[0]; rows=source['rows']; snapshot=rows[0]
                    with self.assertRaises(TypeError): rows[0]=snapshot
                    snapshot['return']=123.0; self.assertNotEqual(rows[0]['return'],123.0)
                    with self.assertRaises(AttributeError): rows.arrays
                    self.assertEqual(source['descriptor'],_read(manifest['folds'][0]['input_manifest']['fold_control']['path'])['evaluation_parts'][0])
                    self.assertEqual(source['header'],_read(source['descriptor']['path']))
                    indexed=_grid(list(rows),lease.common['universe'],source['header']['definition']['sessions'],'feature_session','owner OOS')
                    self.assertEqual(len(indexed),source['header']['row_count'])
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual((state._fixed_shared_bytes,state._fixed_shared_source_bytes),fixed)
                self.assertTrue(all(file_digest(path)==ref and file_fingerprint(path)==fingerprints[path]
                                    for path,ref in records.items()))
                self.assertIsNone(rows._rows)
                with self.assertRaisesRegex(ValueError,'closed'): len(rows)
                with self.assertRaisesRegex(ValueError,'closed'): lease.documents
                self.assertEqual(sorted(hashes),sorted([* _read(runs[0].path/'manifest.json')['files'],'manifest.json']))
                self.assertEqual(len(hashes),9)
                # The default loader remains independent and closes its lease.
                self.assertEqual(load_stock_ml_fold(runs[0].path,batch=batch).identity,runs[0].identity)
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))

    def test_local_pins_exclude_prior_fold_history_in_both_residencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest,runs=self.fixture(Path(temporary))
            for residency in ('eager','sequential'):
                with self.forbidden(),load_stock_ml_batch_inputs(manifest,residency=residency) as batch:
                    state=_data(batch)['matrix_state']
                    with admit_stock_signal_evaluation_fold(self.descriptor(runs[0]),batch=batch) as first:
                        first_pins=dict(first.source_records)
                    with admit_stock_signal_evaluation_fold(self.descriptor(runs[1]),batch=batch) as second:
                        pins=dict(second.source_records); own=manifest['folds'][1]['input_manifest']
                        self.assertIn(own['fold_control']['path'],pins)
                        self.assertNotIn(manifest['folds'][0]['input_manifest']['fold_control']['path'],pins)
                        self.assertFalse(set(str(p) for p in runs[0].path.iterdir()) & set(pins))
                        self.assertTrue(set(str(p) for p in runs[1].path.iterdir()) <= set(pins))
                        control=_read(own['fold_control']['path'])
                        for desc in [*control['raw_parts'],control['normalized'],*control['evaluation_parts']]:
                            self.assertEqual(pins[desc['path']],desc['file_digest'])
                            for buffer in _read(desc['path'])['buffers'].values(): self.assertEqual(pins[buffer['path']],buffer['file_digest'])
                        self.assertIn(manifest['prepared_view']['path'],pins)
                        self.assertIn(_view_data(state.feature)['definition']['source_index']['path'],pins)
                        self.assertTrue(first_pins.keys()-pins.keys())

    def test_error_and_changed_source_release_borrowers(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest,runs=self.fixture(Path(temporary))
            with self.forbidden(),load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; fixed=(state._fixed_shared_bytes,state._fixed_shared_source_bytes)
                with self.assertRaisesRegex(RuntimeError,'consumer failed'):
                    with admit_stock_signal_evaluation_fold(self.descriptor(runs[0]),batch=batch) as lease:
                        rows=lease.evaluation_targets[0]['rows']; raise RuntimeError('consumer failed')
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state.targets,{})
                self.assertIsNone(rows._rows)
                with self.assertRaisesRegex(ValueError,'changed'):
                    with admit_stock_signal_evaluation_fold(self.descriptor(runs[0]),batch=batch) as lease:
                        path=Path(lease.evaluation_targets[0]['header']['buffers']['values']['path'])
                        original=path.read_bytes(); path.write_bytes(original)
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual(state.targets,{})
                self.assertEqual((state._fixed_shared_bytes,state._fixed_shared_source_bytes),fixed)

    def test_descriptor_mismatch_and_pid_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest,runs=self.fixture(Path(temporary))
            with self.forbidden(),load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; descriptor=self.descriptor(runs[0])
                for key in ('file_digest','signal_run_ref'):
                    bad={**descriptor,key:'sha256:'+'0'*64}
                    with self.assertRaisesRegex(ValueError,'descriptor binding'):
                        with admit_stock_signal_evaluation_fold(bad,batch=batch): self.fail('bad descriptor admitted')
                    self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                with admit_stock_signal_evaluation_fold(descriptor,batch=batch) as lease:
                    with patch('axiom_research.stock_signal_evaluation_lease.os.getpid',return_value=-1):
                        with self.assertRaisesRegex(ValueError,'another process'): lease.common

    def test_booster_tamper_and_metadata_budget_failure_leave_no_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest,runs=self.fixture(Path(temporary))
            with self.forbidden(),load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                state=_data(batch)['matrix_state']; fixed=(state._fixed_shared_bytes,state._fixed_shared_source_bytes)
                with patch('axiom_research.stock_signal_evaluation_lease._size',side_effect=ValueError('metadata budget')):
                    with self.assertRaisesRegex(ValueError,'metadata budget'):
                        with admit_stock_signal_evaluation_fold(self.descriptor(runs[0]),batch=batch): self.fail('budget admitted')
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertEqual((state._fixed_shared_bytes,state._fixed_shared_source_bytes),fixed)
                self.assertEqual(state.targets,{})
                (runs[0].path/'booster.txt').write_text('tampered saved model bytes')
                with self.assertRaisesRegex(ValueError,'digest mismatch'):
                    with admit_stock_signal_evaluation_fold(self.descriptor(runs[0]),batch=batch): self.fail('tampered booster admitted')
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))


if __name__=='__main__': unittest.main()

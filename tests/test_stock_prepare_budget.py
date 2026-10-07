"""Declared staging limits on tiny saved inputs; no real data or model run."""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_research import stock_matrix_reader as reader
from axiom_research.stock_batch import load_stock_ml_batch_inputs, _data
from axiom_research.stock_matrix_prepare import prepare_stock_ml_batch_inputs
from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query


class PrepareBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        fixture=PrepareFeatureFixture(cls.root)
        saved=fixture.matrix(); data=PublicDataFixture(fixture.spec)
        module=types.ModuleType('axiom_data')
        module.QuerySpec=Query; module.adjust_prices=data.adjust
        cls.declared=1024**3; cls.forwarded=[]
        native=reader._validate_staged_matrix_batch
        def observed(*args,**kwargs):
            cls.forwarded.append(kwargs['maximum_matrix_bytes'])
            return native(*args,**kwargs)
        try:
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
                 patch.object(reader,'_validate_staged_matrix_batch',side_effect=observed):
                cls.manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,
                    fold_specs=fixture.folds(),destination=cls.root/'prepared',
                    preparation_options={'row_block_sessions':32,'column_block':32,
                        'maximum_resident_bytes':cls.declared,'normalization_backend':'core_cs_batch_v1'})
        finally: saved.close()

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    @contextmanager
    def staging(self):
        target=Path(self.manifest['prepared_view']['path']).parents[1]
        stage=target.parent/'budget-test-stage'
        target.rename(stage)
        try: yield stage,target
        finally: stage.rename(target)

    def admit(self,**limits):
        stores=[]; native=reader.load_matrix_batch_state
        def observed(*args,**kwargs):
            stores.append(kwargs['_store'])
            return native(*args,**kwargs)
        with self.staging() as (stage,target), patch.object(reader,'load_matrix_batch_state',side_effect=observed):
            metrics=reader._validate_staged_matrix_batch(self.manifest,stage=stage,target=target,**limits)
            self.assertFalse(target.exists())
        self.assertEqual(len(stores),1)
        self.assertTrue(stores[0].closed)
        self.assertEqual(metrics['core_calls'],0)
        return stores[0]

    def test_prepare_forwards_exact_saved_budget(self):
        self.assertEqual(self.forwarded,[self.declared])
        self.assertEqual(self.manifest['definition']['preparation_options']['maximum_resident_bytes'],self.declared)

    def test_staged_default_remains_512_mib(self):
        store=self.admit()
        self.assertEqual(store.maximum_matrix_bytes,512*1024**2)
        self.assertEqual(store.maximum_source_bytes,8*1024**3)
        self.assertEqual(store.maximum_parent_bytes,64*1024**2)

    def test_explicit_staged_budget_changes_no_other_limit(self):
        store=self.admit(maximum_matrix_bytes=self.declared)
        self.assertEqual(store.maximum_matrix_bytes,self.declared)
        self.assertEqual(store.maximum_source_bytes,8*1024**3)
        self.assertEqual(store.maximum_parent_bytes,64*1024**2)

    def test_staged_tight_budget_still_rejects(self):
        with self.staging() as (stage,target):
            with self.assertRaisesRegex(ValueError,'budget exceeded'):
                reader._validate_staged_matrix_batch(self.manifest,stage=stage,target=target,maximum_matrix_bytes=1024)
            self.assertFalse(target.exists())

    def test_invalid_explicit_budget_rejects(self):
        with self.staging() as (stage,target):
            for value in (True,False,0,-1,1.5):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError,'positive matrix byte budget'):
                    reader._validate_staged_matrix_batch(self.manifest,stage=stage,target=target,maximum_matrix_bytes=value)

    def test_caller_resident_and_lease_remain_counted(self):
        caller=[0]
        with reader.VerifiedMatrixStore(maximum_matrix_bytes=self.declared,
                _caller_retained_bytes=lambda:caller[0]) as store:
            payload={'unchanged':['small','graph']}
            before=store._native_caller_bytes(payload)
            caller[0]=151; store.resident_bytes=117; store.lease_bytes=131
            self.assertEqual(store._native_caller_bytes(payload)-before,151+117+131)
            store.lease_bytes=self.declared
            with self.assertRaisesRegex(ValueError,'accounting workspace budget exceeded'):
                store._native_caller_bytes(payload)
            store.lease_bytes=0

    def test_public_batch_projection_keeps_real_lease_guards(self):
        batch=load_stock_ml_batch_inputs(self.manifest,
            limits={'maximum_source_bytes':8*1024**3,'maximum_matrix_bytes':self.declared})
        store=_data(batch)['matrix_state'].store; fold=self.manifest['folds'][0]
        try:
            projection=batch._matrix_project(fold['input_manifest'],fold['fold_spec'])
            self.assertGreater(store.lease_bytes,0)
            self.assertEqual(store.borrowers,1)
            with self.assertRaisesRegex(ValueError,'still borrowed'): batch.close()
            projection.close()
            self.assertEqual(store.lease_bytes,0); self.assertEqual(store.borrowers,0)
            store.maximum_matrix_bytes=store.resident_bytes+100
            with self.assertRaisesRegex(ValueError,'budget exceeded'):
                batch._matrix_project(fold['input_manifest'],fold['fold_spec'])
            self.assertEqual(store.lease_bytes,0); self.assertEqual(store.borrowers,0)
        finally: batch.close()


if __name__=='__main__': unittest.main()

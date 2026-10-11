"""Tiny synthetic prepared reuse; no real Data, supplier or account run."""
from pathlib import Path
import builtins
import tempfile
import unittest
from unittest.mock import patch

import test_stock_compact_v3 as fixtures
from axiom_research import (load_stock_feature_view, load_stock_ml_batch_inputs,
    build_stock_ml_fold_from_saved_inputs, load_stock_ml_fold)
from axiom_research.stock_artifacts import file_digest, digest
from axiom_research import stock_compact_labels as owner


class TrainingOptionsTests(unittest.TestCase):
    def test_native_parameter_change_preserves_prepared_and_exact_hit(self):
        helper=fixtures.CompactV3Tests()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=helper.fixture(root)
            with load_stock_feature_view(path) as feature:
                manifest,_=helper.prepare(f,feature,root,folds=f.folds()[:1])
                parents={p:file_digest(p) for p in (root/'prepared').rglob('*') if p.is_file()}
                item=manifest['folds'][0]; stats=[]
                with load_stock_ml_batch_inputs(manifest,feature_inputs=feature) as batch, \
                     patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog):
                    def build(options):
                        metrics={}
                        run=build_stock_ml_fold_from_saved_inputs(item['input_manifest'],
                            fold_spec=item['fold_spec'],destination=root/'models',batch=batch,
                            training_options=options,metrics=metrics)
                        stats.append(metrics); return run
                    baseline=build(None)
                    changed=build({'learning_rate':0.1,'num_boost_round':3})
                    cached=build({'num_boost_round':3,'learning_rate':0.1})
                    self.assertNotEqual(baseline.identity,changed.identity)
                    self.assertEqual(changed.identity,cached.identity); self.assertTrue(cached.reused)
                    self.assertEqual(baseline.to_dict()['dataset_ref'],changed.to_dict()['dataset_ref'])
                    self.assertEqual(baseline.to_dict()['definition']['input_manifest'],item['input_manifest'])
                    self.assertEqual(changed.to_dict()['definition']['input_manifest'],item['input_manifest'])
                    self.assertEqual(changed.model()['parameters']['learning_rate'],0.1)
                    self.assertEqual(changed.model()['num_boost_round'],3)
                    self.assertEqual([(s['train_calls'],s['predict_calls']) for s in stats],[(1,1),(1,1),(0,0)])
                    for s in stats:
                        for counter in ['data_read_calls','supplier_calls','feature_core_calls','label_core_calls','core_calls','account_calls']:
                            self.assertEqual(s[counter],0)
                    with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('load trained')):
                        self.assertEqual(load_stock_ml_fold(changed.path,batch=batch).identity,changed.identity)
                real=builtins.__import__
                def blocked(name,*args,**kwargs):
                    if name.startswith(('axiom_data','axiom_engine','qlib','lightgbm','pandas')):
                        raise AssertionError('readonly runtime import '+name)
                    return real(name,*args,**kwargs)
                with patch('builtins.__import__',side_effect=blocked):
                    self.assertEqual(load_stock_ml_fold(baseline.path).identity,baseline.identity)
                    self.assertEqual(load_stock_ml_fold(changed.path).identity,changed.identity)
                self.assertEqual({p:file_digest(p) for p in parents},parents)

    def test_invalid_options_rejected_before_projection(self):
        invalid=[{'seed':9},{'num_boost_round':True},{'num_boost_round':0},
                 {'num_boost_round':1.5},{'learning_rate':True},{'learning_rate':0},
                 {'learning_rate':float('nan')},{'learning_rate':float('inf')},
                 {'learning_rate':10**1000},[],{'learning_rate':None}]
        for options in invalid:
            with self.subTest(options=options), \
                 patch('axiom_research.stock_matrix_folds._projection',side_effect=AssertionError('projected invalid options')):
                with self.assertRaises(ValueError):
                    build_stock_ml_fold_from_saved_inputs({'contract_version':'stock_ml_saved_inputs_v3'},
                        fold_spec={},destination='unused',training_options=options)

    def test_input_dependencies_exclude_model_code_and_backend_version(self):
        seen=[]
        def version(path):
            seen.append(path.name); return digest(path.name)
        with patch.object(owner,'file_digest',side_effect=version):
            before=owner._implementation()
        self.assertNotIn('stock_matrix_folds.py',seen)
        self.assertNotIn('stock_fold_artifacts.py',seen)
        self.assertNotIn('stock_training.py',seen)
        def model_changed(path):
            return digest('changed') if path.name in {'stock_training.py','stock_matrix_folds.py','stock_fold_artifacts.py'} else digest(path.name)
        with patch.object(owner,'file_digest',side_effect=model_changed):
            self.assertEqual(owner._implementation(),before)
        def input_changed(path):
            return digest('changed') if path.name=='stock_label_contracts.py' else digest(path.name)
        with patch.object(owner,'file_digest',side_effect=input_changed):
            self.assertNotEqual(owner._implementation(),before)
        def metadata(name):
            if name in {'lightgbm','pyqlib'}: raise AssertionError('input queried model backend')
            return 'fixed-'+name
        with patch('importlib.metadata.version',side_effect=metadata):
            self.assertEqual(set(owner._input_environment()['packages']),{'numpy','pandas','pyarrow'})


if __name__=='__main__': unittest.main()

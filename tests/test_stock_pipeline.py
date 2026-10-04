"""One small saved-input fold verifies target semantics and upstream-free reuse."""
import json
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.feature_catalog import load_feature_catalog
from axiom_research.stock_artifacts import digest, write_json, load_stock_ml_experiment
from axiom_research.stock_artifacts import _verify_normalized_labels
from axiom_research.stock_ml import build_stock_ml_experiment, predict_stock_model


def seal(value,key):
    return {**value,key:digest(value)}


class SavedPipelineTests(unittest.TestCase):
    def test_normalized_training_reuses_frozen_inputs_and_native_model(self):
        catalog=load_feature_catalog();selection=[x.to_dict() for x in catalog.default_selection]
        calendar_ref=digest('fixed-calendar')
        symbols=[f'S{i:03}' for i in range(50)]
        calendar=['2024-01-02','2024-01-03','2024-01-04','2024-01-05']
        config=dict(snapshot='s-fixed',universe_id='csi300',symbols=symbols,calendar=calendar,
            read_sessions=calendar,feature_sessions=[calendar[0],calendar[2]],
            prediction_sessions=[calendar[2]],fit_cutoff='2024-01-03T20:30:00Z',
            pit_policy='best_effort_vendor_v1',cutoff_by_session={s:s+'T20:30:00Z' for s in calendar},
            evaluation_cutoff='2024-01-05T20:30:00Z',feature_selection=selection,
            scope_ref=digest('fixed-scope'),calendar_ref=calendar_ref)
        features=seal(dict(contract_version='stock_feature_build_v1',catalog_ref=catalog.identity,
            selection=selection,ordered_features=[x['id'] for x in selection],
            qlib_view={'view_id':'fixed-qlib'},input_evidence_ref=digest([]),rows=[
                dict(security_id=s,session=day,values=[i/50+j for j in range(6)],
                    validity=[True]*6,availability=[day+'T20:00:00Z']*6,reasons=[[]]*6,
                    member=True,knowledge_cutoff=day+'T20:30:00Z',source_refs=['fixed-feature'])
                for day in config['feature_sessions'] for i,s in enumerate(symbols)]),'feature_ref')
        def raw(day,end):
            return seal(dict(contract_version='stock_label_build_v1',calendar_ref=calendar_ref,
                label_spec={'label_id':'forward_5_session_open_close_v1'},rows=[
                    dict(security_id=s,feature_session=day,start_session=end,end_session=end,
                        **{'return':i/1000},label_available_at=end+'T20:00:00Z',valid=True,
                        invalid_reason=None,source_refs=['fixed-raw']) for i,s in enumerate(symbols)]),'label_ref')
        labels=seal(dict(contract_version='stock_label_bundle_v1',training=raw(calendar[0],calendar[1]),
                        evaluation=raw(calendar[2],calendar[3])),'label_ref')
        class NoFacts:
            def __getattr__(self,name): raise AssertionError('Data touched: '+name)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();write_json(old/'feature-inputs.json',[])
            write_json(old/'features.json',features);write_json(old/'labels.json',labels)
            source_doc={'definition':{'config':config,'catalog_ref':catalog.identity},
                        'feature_ref':features['feature_ref'],'label_ref':labels['label_ref']}
            class Saved:
                path=old;identity='fixed-old-experiment'
                def to_dict(self): return source_doc
            def load(path):
                return Saved() if Path(path)==old else load_stock_ml_experiment(path)
            with patch('axiom_research.stock_ml.load_stock_ml_experiment',side_effect=load), \
                 patch('axiom_research.stock_ml._prepare_stock_inputs',side_effect=AssertionError('Feature rerun')):
                metrics={};run=build_stock_ml_experiment(NoFacts(),config=config,destination=root/'new',
                                                       reuse_input_path=old,metrics=metrics)
                self.assertEqual(metrics['data_read_calls'],0)
                self.assertEqual(metrics['feature_core_calls'],0)
                self.assertEqual(metrics['label_core_calls'],2)
                self.assertEqual(metrics['train_calls'],1)
                predictions=run.predictions()
                self.assertEqual(predictions['score_semantics'],'forward_5_session_cs_zscore_prediction')
                self.assertEqual(run.to_dict()['feature_ref'],features['feature_ref'])
                self.assertEqual((run.path/'feature-inputs.json').read_bytes(),(old/'feature-inputs.json').read_bytes())
                bundle=json.loads((run.path/'labels.json').read_text())
                self.assertEqual(bundle['training'],labels['training'])
                values=[r['normalized_target'] for r in bundle['normalized_training']['rows']]
                self.assertAlmostEqual(sum(values)/50,0)
                self.assertAlmostEqual(sum(v*v for v in values)/50,1)
                forged=deepcopy(bundle['normalized_training'])
                forged['sections'][0]['fact_ref']=digest('unrelated-facts')
                forged=seal({k:v for k,v in forged.items() if k!='label_ref'},'label_ref')
                with self.assertRaisesRegex(ValueError,'Core section linkage'):
                    _verify_normalized_labels(forged)
                with patch('lightgbm.train',side_effect=AssertionError('cache trained')), \
                     patch('axiom_research.stock_label_normalization.execute_feature_plan',side_effect=AssertionError('cache Core')):
                    cached_stats={};cached=build_stock_ml_experiment(NoFacts(),config=config,destination=root/'new',
                        reuse_input_path=old,metrics=cached_stats)
                    self.assertTrue(cached.reused);self.assertEqual(cached.identity,run.identity)
                    self.assertEqual(cached_stats['core_calls'],0)
            model=root/'model-only';model.mkdir()
            for name in ('model.json','booster.txt'): shutil.copyfile(run.path/name,model/name)
            rows=[r for r in features['rows'] if r['session']==calendar[2]]
            independent=predict_stock_model(model,rows,ordered_features=features['ordered_features'],feature_selection=selection)
            self.assertEqual(independent,[r['score'] for r in predictions['rows']])
            with self.assertRaisesRegex(ValueError,'semantic versions'):
                predict_stock_model(model,rows,ordered_features=features['ordered_features'],feature_selection=[])


if __name__=='__main__': unittest.main()
